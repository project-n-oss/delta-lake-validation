"""
Performance benchmarking suite.

Operates over 3 datasets:
- nyc_trips (small files)
- nyc_trips (big files)
- synthetic dataset
"""
import time
from functools import wraps
from pyspark.sql import SparkSession, DataFrame, functions, Window


# section: utils


def get_spark_session() -> SparkSession:
    return SparkSession.builder.appName("DatalakeBenchmark").getOrCreate()


def measure_execution_time(func):
    """
    Wrapper function that measures function e2e time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = int((end_time - start_time) * 1000)
        print(f"Function '{func.__name__}' executed in {execution_time} ms")
        return result
    return wrapper


def read_parquet_data(spark, path: str):
    """
    Read one or more parquet files specified at `path`
    """
    return spark.read.parquet(path)


# section: queries for nyc_trips

@measure_execution_time
def nyc_select_0(df):
    print("Selection Query")
    df.select("passenger_count", "trip_distance", "fare_amount").show(2, False)


@measure_execution_time
def nyc_group_by_0(df):
    print("Group by")
    df.groupBy(functions.to_date("tpep_pickup_datetime").alias("pickup_day")).agg(
        functions.sum("total_amount").alias("total_revenue")
    ).show(2, False)


@measure_execution_time
def nyc_filter_0(df):
    # Filtering Query: Filters the dataset to show only the records where the passenger count is greater than 1.
    print("Filter")
    df.filter(df["passenger_count"] > 1).show(2, False)


@measure_execution_time
def nyc_group_by_1(df):
    # Aggregation Query: Calculates and displays the average fare amount and trip distance for the entire dataset.
    print("aggregation")
    df.groupBy().agg(
        functions.avg("fare_amount").alias("average_fare"),
        functions.avg("trip_distance").alias("average_distance")
    ).show(2, False)

    # Date-Time Manipulation: Adds a new column calculating the trip duration in minutes,
    # then filters to show only trips that lasted more than 30 minutes.
    print("Date Time Manipulation")
    df.withColumn("trip_duration",
                  (functions.unix_timestamp("tpep_dropoff_datetime") - functions.unix_timestamp("tpep_pickup_datetime")) / 60
                  ).filter("trip_duration > 30").show(2, False)


@measure_execution_time
def nyc_join_0(df):
    # Create locations_df DataFrame: This section creates a DataFrame containing unique location IDs
    # (PULocationID and DOLocationID) from the original dataset (df). It assigns a dummy location name to
    # each unique location ID, which can be used for more readable data analysis and reporting.
    unique_location_ids = df.select("PULocationID").distinct().union(df.select("DOLocationID").distinct()).distinct()
    locations_df = unique_location_ids.withColumnRenamed("PULocationID", "locationID")
    locations_df = locations_df.withColumn("locationName",
                                           functions.concat(functions.lit("Location "),
                                                            locations_df["locationID"].cast("string")))
    # Join Operation with locations_df: Joins the original dataset with the locations_df DataFrame to enrich
    # the trip data with human-readable location names. This operation is accidentally duplicated in the
    # snippet and can be performed just once.
    print("Join ")
    df.join(locations_df, df.PULocationID == locations_df.locationID).select("trip_distance", "locationName").show(2, False)

    # Window Function: Applies a window function to rank trips by distance within each passenger count group.
    # windowSpec = Window.partitionBy("passenger_count").orderBy(F.desc("trip_distance"))
    # print("Window Function")
    # df.withColumn("rank", F.rank().over(windowSpec)).show(2, False)


@measure_execution_time
def nyc_with_cond_column(df):
    # Conditional Column: Adds a new column indicating whether the tip amount for a trip was high (more than $5).
    print("Conditional Column")
    df.withColumn("high_tip", functions.when(df["tip_amount"] > 5, "Yes").otherwise("No")).show(2, False)


@measure_execution_time
def nyc_with_na_dropped_column(df):
    # Handling Null Values: Removes records from the dataset where critical information (passenger count,
    # trip distance, fare amount) is missing.
    print("handling Null Values")
    df.na.drop(subset=["passenger_count", "trip_distance", "fare_amount"]).show(2, False)


# section: synthetic data [events]

@measure_execution_time
def events_select_0(df):
    """
    Simple select
    """
    df.select("time", "userId", "objectIdStr").show(2, False)


# section: synthetic data [safe events]

@measure_execution_time
def safe_events_select_0(df):
    df.select("user_id", "location_0.latitude").show(2, False)


@measure_execution_time
def safe_events_group_by_0(df):
    df.groupBy("user_id").agg(functions.avg("metrics_0.temperature")).show(3)


@measure_execution_time
def safe_events_group_by_1(df):
    df.groupBy("user_id").agg(functions.avg("metrics_0.temperature")).show(3)


@measure_execution_time
def safe_events_join_0(df, df1):
    # join on safe events user_ids
    df.join(df1, df.user_id == df1.user_id).select(df.event_id, df1.event_id).show(3)


@measure_execution_time
def safe_events_join_1(df, df1):
    # join safe events on bucketed latitude
    df = df.withColumn("location_0.latitude", functions.round("location_0.latitude", 1))
    df1 = df1.withColumn("location_0.latitude", functions.round("location_0.latitude", 1))
    joined = df.join(df1, df.location_0.latitude == df1.location_0.latitude)
    joined.filter(df.user_id != df1.user_id).select(df.user_id, df1.user_id).show(3)


@measure_execution_time
def safe_events_rank(df):
    w = Window.orderBy("participants_0.bio_metrics.calories_burned").partitionBy("participants_0.bio_metrics.heart_rate")
    df.withColumn("drank", functions.rank().over(w)).select("participants_0.bio_metrics.calories_burned", "drank").show(3)


# section: run tests

spark = get_spark_session()

# nyc dataset
# data_location = "gs://<location-of-nyc-dataset>"
# nyc_df = read_parquet_data(spark, data_location)
# nyc_select_0(nyc_df)
# nyc_group_by_0(nyc_df)
# nyc_filter_0(nyc_df)
# nyc_group_by_1(nyc_df)
# nyc_join_0(nyc_df)
# nyc_with_cond_column(nyc_df)
# nyc_with_na_dropped_column(nyc_df)

# se_df = read_parquet_data(spark, "gs://location-of-safe-events-dataset")
# se_df2 = read_parquet_data(spark, "gs://location-of-safe-events-dataset-part2")
# safe_events_select_0(se_df)
# safe_events_group_by_0(se_df)
# safe_events_join_0(se_df, se_df2)
# safe_events_join_1(se_df, se_df2)
# safe_events_rank(se_df)
