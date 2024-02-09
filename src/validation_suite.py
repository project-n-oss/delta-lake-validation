"""
Delta lake validation tests suite. A collections of tests that perform various data and metadata
read/write operations on delta tables.

NOTE: This assumes we are operating on order table (name is parameterized; schema is same)

To run tests (at the bottom of the module):
 1. create TestConfig
object with params with GCS bucket which will store the delta table(s).
 2. uncomment and run test
"""
import pyspark
import time

from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum, auto
from delta.pip_utils import configure_spark_with_delta_pip
from google.cloud import storage
from random import randint, random
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, TimestampType, DoubleType
from typing import List, Union, Optional

# section: constants


GS_URI_PREFIX = "gs://"


# section: types


@dataclass
class OrderRecord:
    order_id: str
    item_count: int
    cost: float
    order_time: datetime
    order_date: date

    @classmethod
    def generate(cls, order_id=None, item_count=None, cost=None, order_time=None, order_date=None):
        """
        Generate record with random values where params are unset
        """
        if order_id is None:
            order_id = f"order_{randint(1, 1000)}"
        if item_count is None:
            item_count = randint(1, 5)
        if cost is None:
            cost = round(random() * 10, ndigits=2)

        # order_data and order_time must be "aligned"
        if order_time is None and order_date is None:
            order_date = date(2024, 1, randint(1, 30))
            order_time = datetime(order_date.year, order_date.month, order_date.day, randint(0, 23), randint(0, 59))
        elif order_date is None:
            order_date = datetime_to_date(order_time)
        elif order_time is None:
            order_time = datetime(order_date.year, order_date.month, order_date.day, randint(0, 23), randint(0, 59))
        else:
            assert datetime_to_date(order_time) == order_date, "order_date and order_time are not aligned"

        return cls(order_id, item_count, cost, order_time, order_date)


class TestRunMode(Enum):
    """
    How to run the tests.
    In particular, some tests need to validate interleaved
    behavior, e.g. delta op, crunch, delta op. In this case,
    test can be run first with `setup_only` and then with `validation_only`.
    Otherwise, tests will run both the setup and the validation `setup_and_validate` in one step
    """
    setup_only = auto()
    validation_only = auto()
    setup_and_validate = auto()


@dataclass
class TestConfig:
    """
    Contains config for tests
    Additionally contains methods to facilitate running tests
    """
    # name and location of main table
    table_name: str
    table_location: str
    # name and location of updates table
    updates_table_name: str
    updates_table_location: str

    # run params
    # whether to recreate table(s) before running tests
    recreate_setup: bool = True
    exec_mode: TestRunMode = TestRunMode.setup_and_validate
    # whether to use a catalog
    catalog_enabled: bool = False

    def get_table_id(self) -> str:
        """
        delta tables can be identified by:
        1) (multi-part) name, e.g. "schema.table"
        2) by location, e.g. "delta.`gs://bucket/root/`"
        """
        if self.catalog_enabled:
            return self.table_name
        else:
            return table_id_from_location(self.table_location)

    def get_updates_table_id(self) -> str:
        if self.catalog_enabled:
            return self.updates_table_name
        else:
            return table_id_from_location(self.updates_table_location)

# section: utils

def table_id_from_location(table_location: str) -> str:
    """
    Get delta table identifier from location
    """
    return f"delta.`{table_location}`"


def to_hive_path(order_date: date) -> str:
    """
    Convert order_date to
    """
    return f"order_date={to_date_literal(order_date)}"


def datetime_to_date(dt: datetime) -> date:
    return date(dt.year, dt.month, dt.day)


def to_datetime_literal(dt: Union[date, datetime]) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def prune_trailing_slash(path: str) -> str:
    if len(path) > 0 and path[-1] == "/":
        return path[:-1]
    return path


def to_date_literal(dt: Union[date, datetime]) -> str:
    return dt.strftime("%Y-%m-%d")


def get_orders_schema():
    return StructType([
        StructField("order_id", StringType(), True),
        StructField("item_count", IntegerType(), True),
        StructField("cost", DoubleType(), True),
        StructField("order_time", TimestampType(), True),
        StructField("order_date", DateType(), True)
    ])


def list_objects(root_dir_uri: str) -> List:
    """
    List all objects in bucket starting at given prefix.

    root_dir_uri: e.g. gs://<bucket>/<path/to/root/of/dir>
    exclude_dir: exclude entries that end with trailing slash
    """
    if not root_dir_uri.startswith(GS_URI_PREFIX):
        raise ValueError("expected path start with: [gs://]")

    # determine if there is a prefix or start at bucket root
    dir_path = root_dir_uri[len(GS_URI_PREFIX):]
    idx = dir_path.find("/")
    if idx == -1:
        bucket_name = dir_path
        prefix = None
    else:
        bucket_name = dir_path[:idx]
        prefix = dir_path[idx+1:] or None

    client = storage.Client()

    # get objects
    bucket = client.get_bucket(bucket_name)
    if prefix is None:
        blobs = bucket.list_blobs()
    else:
        blobs = bucket.list_blobs(prefix=prefix)

    # filter directory prefixes
    objects = []
    for blob in blobs:
        if not blob.name.endswith("/"):
            objects.append(blob)

    return objects


def get_partition_files(table_location: str, order_date: date) -> List[str]:
    partition_uri = f"{prune_trailing_slash(table_location)}/{to_hive_path(order_date)}"
    blobs = list_objects(partition_uri)
    objects = []
    for blob in blobs:
        objects.append(blob.name)
    return objects


def delete_bucket_objects(root_dir_uri: str, dry_run=True):
    """
    List and delete all objects in bucket at given prefix.

    root_dir_uri := gs://<bucket>/<path/to/root/of/deletion>
    """
    blobs = list_objects(root_dir_uri)
    for blob in blobs:
        if dry_run:
            print(f"would delete object: {blob.name}")
        else:
            print(f"deleting object: {blob.name}")
            blob.delete()


# section: spark util functions

def get_session():
    """
    get spark session
    """
    # NOTE(dataproc): the configs will only take effect if they were set at cluster creation time
    builder = pyspark.sql.SparkSession.builder.appName("MyApp") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    return spark


# section: delta metadata/utility ops


def list_tables(spark):
    spark.sql("SHOW TABLES").show()


def desc_table(spark, table_name: str):
    spark.sql(f"DESCRIBE DETAIL {table_name}").show()


def show_history(spark, table_name: str):
    spark.sql(f"DESCRIBE HISTORY {table_name}").show()


def get_history(spark, table_name: str) -> List[dict]:
    """
    Return list of dicts corresponding to different versions of table.
    Contains fields: "version", "timestamp", "operation".
    History is ordered reverse chronologically
    """
    history = spark.sql(f"DESCRIBE HISTORY {table_name}").collect()
    return [record.asDict() for record in history]


def drop_table(spark, table_name: str):
    sql = f"DROP TABLE IF EXISTS {table_name}"
    print(f"Running sql: {sql}")
    spark.sql(sql).show()


def create_table(spark, table_name: str, table_location: str):
    """
    Create delta table with orders schema, parameterized by `table_name` and `table_locations`
    """
    sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        order_id STRING, 
        order_time TIMESTAMP, 
        item_count INT, 
        cost DOUBLE, 
        order_date DATE
    )
    USING DELTA
    LOCATION '{table_location}'      
    PARTITIONED BY (order_date)
    TBLPROPERTIES(delta.enableChangeDataFeed = true)
    """
    print(f"Running sql: {sql}")
    spark.sql(sql).show()


def register_table(spark, table_name: str, table_location: str):
    """
    register table that already exists at location, i.e. created without metastore
    """
    sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} 
    USING DELTA
    LOCATION '{table_location}'      
    """
    print(f"Running sql: {sql}")
    spark.sql(sql).show()


def enable_change_data_feed(spark, table_name: str):
    spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")


def recreate_table(spark, table_name: str, table_location: str, data: Optional[List] = None):
    """
    Assume `orders` schema.
    Recreate named table by:
        - delete existing table
        - write some data (in delta format) to location
        - and register table

    If user does not provide `data`, then mock data is generated to create table;
    and subsequently deleted
    """
    # drop table - only removes tables from catalog
    drop_table(spark, table_name)
    # delete data - otherwise recreating table at same location, will have old data
    delete_bucket_objects(table_location, dry_run=False)

    # get seed data
    seed_date = date(2000, 1, 1)
    if not data:
        seed_data = [OrderRecord.generate(order_date=seed_date)]
    else:
        seed_data = data

    # write some data
    write_data(spark, table_location, seed_data)

    # register table
    register_table(spark, table_name, table_location)

    # enable change data feed
    enable_change_data_feed(spark, table_name)

    if data is None:
        # delete data if seed data was not caller provided
        seed_data_date = to_date_literal(seed_date)
        sql = f'DELETE FROM {table_name} WHERE order_date = "{seed_data_date}"'
        spark.sql(sql)


def recreate_table_v2(spark, table_name: str, table_location: str):
    drop_table(spark, table_name)
    delete_bucket_objects(table_location, dry_run=False)
    create_table(spark, table_name, table_location)


# section: test related ops


def write_data(spark, location: str, data: List, write_mode="append"):
    """
    write to delta table.
    NOTE: this writes to the location
    write_mode: ("append", "overwrite")
    """
    order_schema = get_orders_schema()
    df = spark.createDataFrame(data, schema=order_schema)
    df.write.partitionBy("order_date").format("delta").mode(write_mode).save(location)

    
def insert_records(spark, table_name: str, records: List[OrderRecord]):
    """
    Insert(append) records to table.
    Inserts are batched by partition_date

    Example stmt:   
    INSERT INTO orders PARTITION (order_date =  date'2024-01-01') (order_id, item_count, cost, order_time) VALUES
        ('order_0', 1, cast('10.0' as double), timestamp'2024-01-01 16:16:16'),
        ('order_1', 1, cast('10.0' as double), timestamp'2024-01-01 16:16:16');
    """
    # construct batches
    batches = {}
    for record in records:
        if record.order_date not in batches:
            batches[record.order_date] = []
        batches[record.order_date].append(record)
    
    # convert each batch into an insert statement
    for batch_date, batch_records in batches.items(): 
        head = f"""INSERT INTO {table_name} PARTITION 
                    (order_date =  date'{to_date_literal(batch_date)}') 
                    (order_id, item_count, cost, order_time) VALUES \n"""
        rows_sql = []
        for record in batch_records:
            row_sql = f"('{record.order_id}', {record.item_count}, cast('{record.cost}' as double), timestamp'{to_datetime_literal(record.order_time)}')"
            rows_sql.append(row_sql)
        tail = ', \n'.join(rows_sql)
        sql = head + tail
        # print(f"Running sql: {sql}")
        spark.sql(sql)


def read_parquet_file(spark, file_location: str):
    """
    Read parquet file at `file_location` (full-qualified GCS location).
    And print count
    """
    df = spark.read.format("parquet").load(file_location)
    print("count", df.count())


def read_table(spark, table_name: str):
    """
    Read data using table_name.
    """
    sql = f"SELECT count(*) FROM {table_name}"
    print(f"Running sql: {sql}")
    spark.sql(sql).show()

    sql = f"SELECT * FROM {table_name}"
    print(f"Running sql: {sql}")
    spark.sql(sql).show()


def read_table_from_location(spark, location: str):
    """
    Read data using location.
    """
    spark.sql(f"SELECT count(*) FROM delta.`{location}`").show()
    spark.sql(f"SELECT * FROM delta.`{location}`").show()


def read_table_from_time(spark, table_name: str, time_travel_point: Union[date, datetime]):
    """
    Query the table from a different time point
    """
    date_lit = to_datetime_literal(time_travel_point)
    sql = f'SELECT count(*) FROM {table_name} TIMESTAMP AS OF "{date_lit}"'
    print(f"Running sql: {sql}")
    spark.sql(sql).show()

    sql = f'SELECT * FROM {table_name} TIMESTAMP AS OF "{date_lit}"'
    print(f"Running sql: {sql}")
    spark.sql(sql).show()


def read_table_from_version(spark, table_name: str, version="2"):
    """
    Query a different version of table
    """
    sql = f"SELECT count(*) FROM {table_name} VERSION AS OF {version}"
    print(f"Running sql: {sql}")
    spark.sql(sql).show()

    sql = f"SELECT * FROM {table_name} VERSION AS OF {version}"
    print(f"Running sql: {sql}")
    spark.sql(sql).show()


def vacuum_table(spark, table_id: str):
    """
    vacuum table
    """
    spark.sql("SET spark.databricks.delta.retentionDurationCheck.enabled=false")
    sql = f"VACUUM {table_id} RETAIN 0 HOURS"
    print(f"Running: {sql}")
    spark.sql(sql).show()


def optimize_table(spark, table_id: str):
    """
    Optimize table.

    forms:
        OPTIMIZE TABLE_NAME
        OPTIMIZE TABLE_NAME WHERE order_date = '2024-01-01'
        OPTIMIZE TABLE_NAME WHERE order_date = '2024-01-01' ZORDER BY order_id
    """
    sql = f"OPTIMIZE {table_id}"
    print(f"Running: {sql}")
    spark.sql(sql).show()


def read_change_data_feed(spark, table_id: str, version_num="0"):
    """
    read change data feed
    """
    sql = f"SELECT * FROM table_changes('{table_id}', {version_num})"
    print(f"Running sql: {sql}")
    spark.sql(sql).show()


def merge_tables(spark, table_name: str, updates_table_name: str):
    """
    Merge tables.
    Merge when order_ids match. With values being updated to the updates table
    """
    # merge main and updates table
    # if conflicted, updates win
    sql = f"""
    MERGE INTO {table_name}
    USING {updates_table_name}
    ON {updates_table_name}.order_id = {table_name}.order_id
    WHEN MATCHED THEN
        UPDATE SET
        order_id = {updates_table_name}.order_id,
        order_time = {updates_table_name}.order_time, 
        item_count = {updates_table_name}.item_count,
        cost = {updates_table_name}.cost, 
        order_date = {updates_table_name}.order_date       
    WHEN NOT MATCHED
        THEN INSERT (
            order_id,
            order_time,
            item_count,
            cost,
            order_date
    )
    VALUES (
        {updates_table_name}.order_id,
        {updates_table_name}.order_time,
        {updates_table_name}.item_count,
        {updates_table_name}.cost,
        {updates_table_name}.order_date
    )
    """
    print(f"Running merge sql: {sql}")
    spark.sql(sql).show()


def setup_tables(spark, config: TestConfig, setup_updates_table = False):
    """
    (Re)Create delta tables
    """
    if config.recreate_setup:
        if config.catalog_enabled:
            recreate_table(spark, config.table_name, config.table_location)
            if setup_updates_table:
                recreate_table(spark, config.table_name, config.table_location)
        else:
            delete_bucket_objects(config.table_location, dry_run=False)
            if setup_updates_table:
                delete_bucket_objects(config.updates_table_location, dry_run=False)



# section : test suite
    
def test_1_write_read_to_delta(spark, config: TestConfig):
    """
    Test 1: write to delta table
    Expected output: should print inserted rows        
    """
    print("Running Test 1: writing and reading delta lake")
    setup_tables(spark, config)

    data = [
        OrderRecord.generate(order_id="order_0"),
        OrderRecord.generate(order_id="order_1"),
    ]
    write_data(spark, config.table_location, data)
    read_table(spark, config.get_table_id())


def test_2_time_travel_read(spark, config: TestConfig):
    """
    Test 2: read from time travel point. 
    Setup: Insert some data at time t0, busy wait, insert more data at time t1.
    time travel to (t1 - epsilon) and read data and only data from. 

    Expected output: should print only rows inserted in first batch
    """
    print("Running Test 2: time travel read")
    setup_tables(spark, config)

    data = [
        OrderRecord.generate(order_id="order_0_before_time_travel", order_date=date(2024,1,10)),
        OrderRecord.generate(order_id="order_1_before_time_travel", order_date=date(2024,1,10))
    ]
    write_data(spark, config.table_location, data)

    # get write timestamp from delta history
    history = get_history(spark, config.get_table_id())
    time_travel_point = history[0]["timestamp"]

    # ensure writes don't have same timestamp
    time.sleep(1)

    data = [
        OrderRecord.generate(order_id="order_0_after_time_travel", order_date=date(2024,1,10)),
        OrderRecord.generate(order_id="order_1_after_time_travel", order_date=date(2024,1,10))
    ]
    write_data(spark, config.table_location, data)
    print("Reading full table; we expect to see all [4] records")
    read_table(spark, config.get_table_id())

    print("Reading table from time stamp; we should see partial [2] records")
    read_table_from_time(spark, config.get_table_id(), time_travel_point)


def test_3_read_table_version(spark, config: TestConfig):
    """
    Test 3: read table from a given version

    Expected output: should print only rows inserted in first batch
    """
    print("Running Test 3: versioned read")
    setup_tables(spark, config)

    data = [
        OrderRecord.generate(order_id="order_0_before_version", order_date=date(2024,1,10)),
        OrderRecord.generate(order_id="order_1_before_version", order_date=date(2024,1,10))
    ]
    write_data(spark, config.table_location, data)

    # get table version
    history = get_history(spark, config.get_table_id())
    version = history[0]["version"]

    data = [
        OrderRecord.generate(order_id="order_0_after_version", order_date=date(2024,1,10)),
        OrderRecord.generate(order_id="order_1_after_version", order_date=date(2024,1,10))
    ]
    write_data(spark, config.table_location, data)
    print("Reading full table; we expect to see all [4] records")
    read_table(spark, config.get_table_id())

    print(f"Reading table from version={version}; we should see partial [2] records")
    read_table_from_version(spark, config.get_table_id(), version=version)


def test_4_read_change_data_feed(spark, config: TestConfig):
    """
    Read table change data feed.
    Expected outcome: show change feed
    """
    print("Running Test 6: Read change feed")
    setup_tables(spark, config)
    read_change_data_feed(spark, config.get_table_id(), version_num="1")



def test_5_merge_data(spark, config: TestConfig):
    """
    Test merge operation.
    Setup:
    - Write data to main table
    - write conflicting data to updates table
    - merge with policy that updates table win
    Expected outcome: main table has updated records
    """
    print("Running Test 5: Merge")
    setup_tables(spark, config, setup_updates_table=True)

    # populate main table
    data = [
        OrderRecord.generate(order_id="order_0", item_count=10),
        OrderRecord.generate(order_id="order_1", item_count=20)
    ]
    write_data(spark, config.table_location, data)

    # populate updates table with conflicting records
    data[0].item_count = 111
    data[1].item_count = 222
    write_data(spark, config.updates_table_location, data)

    print("Printing records [main table]")
    read_table(spark, config.get_table_id())

    print("Printing records [updates table] [pre-merge]")
    read_table(spark, config.get_updates_table_id())

    merge_tables(spark, config.get_table_id(), config.get_updates_table_id())

    print("Printing records [updates table] [post-merge]")
    print("Expected record with order_id='order_0' with item_count=111")
    print("Expected record with order_id='order_1' with item_count=222")
    read_table(spark, config.get_updates_table_id())


def test_6_overwrite_data(spark, config: TestConfig):
    """
    Test overwrite.
    Setup:
        - insert data
        - overwrite data
        - read data
    Expected outcome: only new data should exist
    """
    print("Running Test 6: Overwrite")
    setup_tables(spark, config)

    data = [
        OrderRecord.generate(order_id="order_0"),
        OrderRecord.generate(order_id="order_1")
    ]
    write_data(spark, config.table_location, data)
    print("Reading table before overwrite")
    read_table(spark, config.get_table_id())

    # mutate data
    data = [
        OrderRecord.generate(order_id="order_999", item_count=10),
        OrderRecord.generate(order_id="order_888", item_count=20)
    ]
    write_data(spark, config.table_location, data, write_mode="overwrite")
    print("Reading table after overwrite")
    read_table(spark, config.get_table_id())


def test_7_delete_data(spark, config: TestConfig):
    """
    Tests Delete op
    Expected outcome: deleted data should not exist
    """
    print("Running Test 7: Delete")

    setup_tables(spark, config)
    to_delete_date = date(year=2024, month=1, day=10)
    # data for these partitions [16, 16] will be kept
    to_keep_date = date(year=2024, month=1, day=16)
    # number of records per partition
    records_per_partition = 5

    # 2. setup test
    to_delete_records = [OrderRecord.generate(order_date=to_delete_date) for _ in range(records_per_partition)]
    print(f"Inserting records [to delete]: [count: {len(to_delete_records)}] {to_delete_records}")
    write_data(spark, config.table_location, to_delete_records)

    to_keep_records = [OrderRecord.generate(order_date=to_keep_date) for _ in range(records_per_partition)]
    print(f"Inserting records [to live]: [count: {len(to_keep_records)}] {to_keep_records}")
    write_data(spark, config.table_location, to_keep_records)

    # 3. validate
    print("Reading full table [pre-delete]. We expect [10] records")
    read_table(spark, config.get_table_id())
    order_date_literal = to_date_literal(date(2024, 1, 10))
    sql = f'DELETE FROM {config.get_table_id()} WHERE order_date = "{order_date_literal}"'
    print(f"Running sql: [{sql}]")
    spark.sql(sql)
    print(f"Reading full table [post-delete]. We expect [5] records for [{to_keep_date}]")
    read_table(spark, config.get_table_id())


def test_8_update_data(spark, config: TestConfig):
    """
    Test update op.
    Setup:
        - insert records
        - update records (records with even item counts are set to 0)
        - read table
    Expected outcome: records should be updated
    """
    print("Running Test 8: Update table")

    records = [OrderRecord.generate(item_count=count) for count in range(1, 10)]
    write_data(spark, config.table_location, records)
    print("Reading table [pre-update]")
    read_table(spark, config.get_table_id())

    # update all even item_count values to 0
    sql = f"UPDATE {config.get_table_id()} SET item_count = 0 WHERE MOD(item_count, 2) = 0"
    print(f"Running sql: {sql}")
    spark.sql(sql)

    print("Reading table [post-update]")
    read_table(spark, config.get_table_id())


def test_9_vacuum_table(spark, config: TestConfig):
    """
    Test vacuum.
    Setup:
    - Insert data in many partitions.
    - Delete data in predetermined partitions
    - vacuum

    Expected outcome: no data files in deleted partition dir

    NOTE: This vacuums relies on objects being physically partitioned (named) based on hive partitioning scheme.
    Which is not required by delta spec. But if the objects are not hive partitioned this test is invalid.
    """
    print("Running Test 9: Vacuum test")

    # 1. recreate table
    setup_tables(spark, config)

    # data for these partitions [10, 10] will be deleted
    to_delete_date = date(year=2024, month=1, day=10)
    # data for these partitions [16, 16] will be kept
    to_keep_date = date(year=2024, month=1, day=16)
    # number of records per partition
    records_per_partition = 5

    # 2. setup test
    if config.exec_mode == TestRunMode.setup_only or config.exec_mode == TestRunMode.setup_and_validate:
        to_delete_records = [OrderRecord.generate(order_date=to_delete_date) for _ in range(records_per_partition)]
        print(f"Inserting records [to delete]: [count: {len(to_delete_records)}] {to_delete_records}")
        write_data(spark, config.table_location, to_delete_records)

        to_keep_records = [OrderRecord.generate(order_date=to_keep_date) for _ in range(records_per_partition)]
        print(f"Inserting records [to live]: [count: {len(to_keep_records)}] {to_keep_records}")
        write_data(spark, config.table_location, to_keep_records)
    else:
        print(f"Skipping test setup [config.exec_mode = {config.exec_mode}].")

    if config.exec_mode == TestRunMode.setup_only:
        # 2.1. exit; setup only run
        return

    # 3. run validation
    print("Reading full table, expect 10 records")
    read_table(spark, config.get_table_id())

    # 3.1. perform delete
    lower_bound = to_datetime_literal(date(2024, 1, 10))
    upper_bound = to_datetime_literal(date(2024, 1, 10))
    sql = f'DELETE FROM {config.get_table_id()} WHERE order_date >= "{lower_bound}" AND order_date <= "{upper_bound}"'
    print(f"Running: {sql}")
    spark.sql(sql)

    data_files = get_partition_files(config.table_location, to_delete_date)
    print(f"[pre-vacuum] Found [{len(data_files)}] data files for to be deleted partition [{to_delete_date}]; expected > 0")

    # 3.2. perform vacuum
    vacuum_table(spark, config.get_table_id())

    # 3.3. check partition
    data_files = get_partition_files(config.table_location, to_delete_date)
    print(f"[post-vacuum] Found [{len(data_files)}] data files for deleted partition [{to_delete_date}]; expected == 0")
    data_files = get_partition_files(config.table_location, to_keep_date)
    print(f"[post-vacuum] Found [{len(data_files)}] data files for kept partition [{to_keep_date}]; expected > 0")


def test_10_optimize_table(spark, config: TestConfig):
    """
    Test 5: test optimize table command

    Optimize performs multiple ops, including:
        - compaction of multiple small files into fewer files (exercized in this test)
        - remove logically dropped columns

    Setup:
        - perform many small inserts to create many small data files
        - read file/objects before optimize
        - optimize
        - read file/objects after optimize

    Expected outcome: new data files after running optimize
    NOTE: old data files won't be removed until a "vacuum" is run.
    """
    print("Running Test 10: Optimize table")

    # 1. table setup
    setup_tables(spark, config)

    # 2. validation setup
    # insert N batches of size M
    batch_count = 5
    batch_size = 5000
    reference_date = date(2024, 1, 1)
    for batch_num in range(batch_count):
        records = [OrderRecord.generate(order_date=reference_date) for _ in range(batch_size)]
        print(f"Inserting [count: {len(records)}] records [batch: {batch_num}]")
        write_data(spark, config.table_location, records)

    old_files = set(get_partition_files(config.table_location, reference_date))

    # optimize
    optimize_table(spark, config.get_table_id())

    new_files = set(get_partition_files(config.table_location, reference_date))
    only_new = new_files.difference(old_files)
    only_old = old_files.difference(new_files)

    # we expect optimize to have created a new file(s)
    print(f"Total number of files: [before optimize = {len(old_files)}, after optimize = {len(new_files)}]")
    print(f"Number of files *only* present before optimize [count: {len(only_old)}]; we expect == 0")
    print(f"Number of files *only* present after optimize [count: {len(only_new)}]; we expect > 0")
    print(f"New data files: [{only_new}]")



def test_11_concurrent_writes_same_partition(spark, config: TestConfig):
    """
    NOTE: This test must be manually triggered from multiple workers simultaneously
    Expected outcome: writes should fail
    See: https://docs.delta.io/latest/concurrency-control.html#id3
    """
    print("Running Test 9: Concurrent write to the same partition")
    order_date = date(2024, 1, 1)
    records = [OrderRecord.generate(order_date=order_date) for _ in range(10)]
    print(f"Writing records: {records}")
    write_data(spark, config.table_location, records)
    write_data(spark, config.table_location, records, write_mode="overwrite")


def test_12_concurrent_writes_different_partition(spark, config: TestConfig):
    """
    NOTE: This test must be manually triggered from multiple workers simultaneously.
    Expected outcome: writes should succeed
    """
    print("Running Test 10: Concurrent write to the different partition")
    order_dates = [date(2024, 1, 1), date(2024, 1, 2)]
    records = [OrderRecord.generate(order_date=order_dates[0]) for _ in range(5)]
    records.extend([OrderRecord.generate(order_date=order_dates[1]) for _ in range(5)])
    print(f"Writing records: {records}")
    write_data(spark, config.table_location, records)


# section: run tests

# step 1: define config
test_config = TestConfig(
    table_location = "gs://path/to/table/root",
    table_name = "orders",
    updates_table_name = "orders_updates",
    updates_table_location = "gs://path/to/updates/table/root",
    recreate_setup=True,
    exec_mode=TestRunMode.setup_and_validate,
    catalog_enabled=False,
)

# only needed locally
# spark = get_session()

# sanity check
#read_parquet_file(spark, "gs://path/to/table/path/to/object")

# step 2: run tests
# read tests
test_1_write_read_to_delta(spark, test_config)
#test_2_time_travel_read(spark, test_config)
#test_3_read_table_version(spark, test_config)
#test_4_read_change_data_feed(spark, test_config)

# write tests
#test_5_merge_data(spark, test_config)
#test_6_overwrite_data(spark, test_config)
#test_7_delete_data(spark, test_config)
#test_8_update_data(spark, test_config)

# write tests - metadata
#test_9_vacuum_table(spark, test_config)
#test_10_optimize_table(spark, test_config)

# concurrency tests
#test_11_concurrent_writes_same_partition(spark, test_config) # 11
#test_12_concurrent_writes_different_partition(spark, test_config) # 12
