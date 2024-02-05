# Functional Validation for Delta Lake

## Overview

This contains a suite of tests that perform various data and metadata read/write operations on delta tables.
This includes operations to:
- create table
- write to table
  - append
  - overwrite
  - update
  - merge
- read a table
  - read different version of table
  - read table from a different time
- read a table's change delta feed
- delete records from a table
- vacuum
- optimize
  - bin-packing
  - z-ordering
- operations run concurrently

## Test Setup

### Compute
The suite can be run anywhere there is pyspark, and other dependencies.
The dependencies are specified in `pyproject.toml`.

### Storage
The current setup has been tested with Google Cloud Storage (GCS)
The suite has GCS specific utilities, e.g. to read the object store to validate certain operations
However, the actual Delta Lake tests are agnostic to the underlying storage.

## Running Tests

The test suite is a single module, with a config at the bottom of the module, i.e.
```
test_config = TestConfig(
    table_location = "gs://path/to/table/root",
    table_name = "orders",
    updates_table_name = "orders_updates",
    updates_table_location = "gs://path/to/updates/table/root",
    recreate_setup=True
)
```
1. Update `table_location = "gs://path/to/table/root"` and `updates_table_location = "gs://path/to/updates/table/root"`
to appropriate locations. 

Note: These locations should be empty. Any objects located in the "tree" rooted at the either location will be deleted.

2. Uncomment tests to run (in the module at the bottom). 

3. The test suite (src/delta_validation/) can be copied as-is and run in: jupyter notebook, pyspark job etc.


### Expected outcome

The expected outcomes for each test are specified in-line in the test definitions.
