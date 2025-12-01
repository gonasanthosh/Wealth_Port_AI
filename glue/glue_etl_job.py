# glue/glue_etl_job.py
# Robust Glue PySpark script: reads CSV(s) from --source_s3, writes parquet to --target_s3/users/
# Uses recursiveFileLookup and logs read/write counts.
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql.functions import col
import sys, logging, os

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# arguments: Glue always provides JOB_NAME; we expect --source_s3 and --target_s3
args_keys = ['JOB_NAME','source_s3','target_s3']
args = getResolvedOptions(sys.argv, args_keys)

sc = SparkContext.getOrCreate()
glue_ctx = GlueContext(sc)
spark = glue_ctx.spark_session

source = args['source_s3']
target = args['target_s3']

logger.info(f"Glue ETL starting. Source: {source}  Target: {target}")

# Read CSV files robustly (header + recursive lookup)
try:
    df = spark.read.option('header','true').option('recursiveFileLookup','true').csv(source)
except Exception as e:
    logger.exception("Error reading source CSVs")
    raise

logger.info(f"Schema read: {df.schema.simpleString()}")
try:
    count_read = df.count()
except Exception as e:
    logger.exception("Error while counting rows")
    count_read = 0

logger.info(f"Rows read from source: {count_read}")

# Basic cleansing and type casting
if 'user_id' in df.columns:
    df2 = df.dropna(subset=['user_id']).dropDuplicates(['user_id']).withColumn('user_id', col('user_id').cast('int'))
else:
    df2 = df

count_after = df2.count()
logger.info(f"Rows after cleansing/dedup: {count_after}")

if count_after == 0:
    logger.warning("No rows to write. Exiting without writing parquet.")
else:
    out_path = os.path.join(target.rstrip('/'), 'users/')
    logger.info(f"Writing parquet to: {out_path}")
    # coalesce to 1 file for small datasets (remove coalesce in production)
    try:
        df2.coalesce(1).write.mode('overwrite').parquet(out_path)
        logger.info("Parquet write completed.")
    except Exception as e:
        logger.exception("Error writing parquet")
        raise

logger.info("Glue ETL finished.")
