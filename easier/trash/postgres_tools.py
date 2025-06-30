import os
import easier as ezr
import ibis
from .constants import PRODUCTION_CONN_NAME, DOCKERPG_CONN_NAME


def get_postgres_creds(name):
    allowed_names = [PRODUCTION_CONN_NAME, DOCKERPG_CONN_NAME]
    if name == "production":
        kwargs = {
            "host": os.environ["PGHOST"],
            "user": os.environ["PGUSER"],
            "password": os.environ["PGPASSWORD"],
            "dbname": os.environ["PGDATABASE"],
        }
    elif name == "docker":
        kwargs = {
            "host": "db",
            "user": "postgres",
            "password": "postgres",
            "dbname": "postgres",
        }
    else:
        raise ValueError(f"{name} not in {allowed_names}")

    if not kwargs.get("host"):
        raise ValueError(f"No credentials for postgres connection name {name}")
    return kwargs


def get_postgres_ibis_connection(name):
    kwargs = get_postgres_creds(name)
    kwargs["database"] = kwargs.pop("dbname")
    conn = ibis.postgres.connect(**kwargs)
    return conn


# def get_postgres_query_obj(name):
#     kwargs = get_postgres_creds(name)
#     pg = ezr.PG(**kwargs)
#     return pg


# def get_postgres_ibis_connections():
#     dev_mode =  os.environ.get('DEV_MODE', 'FALSE')
#     if dev_mode == 'TRUE':
#         url = ezr.pg_creds_from_env(force_docker=True)
#     else:
#         url = ezr.pg_creds_from_env()
#     conn = ibis.postgres.connect(url=url)
#     return conn


# def create_postgres_functions():
#     # Get a connection to the bodhi production db
#     pg = get_postgres_query_obj('production')

#     # Create functions that show currently defined postgres functions
#     pg.query("""

#     -------------------------------------------------------------------
#     -------------------------------------------------------------------
#     CREATE OR REPLACE FUNCTION
#         bodi_list_functions()

#     RETURNS
#         table (
#             function_schema varchar,
#             function_name varchar
#         )
#     LANGUAGE plpgsql

#     AS $$
#     BEGIN
#         RETURN QUERY
#             SELECT
#                 n.nspname::varchar AS function_schema,
#                 p.proname::varchar AS function_name
#             FROM
#                 pg_proc p
#                 LEFT JOIN pg_namespace n ON p.pronamespace = n.oid
#             WHERE
#                 --n.nspname NOT IN ('pg_catalog', 'information_schema')
#                 p.proname LIKE 'bodi%'
#             ORDER BY
#                 function_schema,
#                 function_name;
#     END;$$;

#     -------------------------------------------------------------------
#     -------------------------------------------------------------------
#     CREATE OR REPLACE FUNCTION
#         bodi_get_function_code(
#             function_name varchar
#         )

#     RETURNS
#         table (
#             code varchar
#         )
#     LANGUAGE plpgsql

#     AS $$
#     BEGIN

#         RETURN QUERY
#             SELECT
#                 routine_definition::varchar
#             FROM
#                 information_schema.routines
#             WHERE
#                 routine_name::varchar = function_name;

#     END;
#     $$
#     """)
#     pg.run()

#     # Create a function that gets production history for homes.
#     pg.query("""
#         -------------------------------------------------------------------
#         -------------------------------------------------------------------
#         CREATE OR REPLACE FUNCTION
#             bodi_get_raw_history(
#                 starting timestamp,
#                 ending timestamp,
#                 production_threshold double precision
#             )

#         RETURNS
#             table (
#                 homeowner_id integer,
#                 date timestamp,
#                 total_production double precision
#             )
#         LANGUAGE plpgsql

#         AS $$
#         BEGIN
#         RETURN QUERY
#             SELECT
#                 hist."homeownerId" AS homeowner_id,
#                 hist.date::timestamp,
#                 hist."totalProduction" as total_production
#             FROM
#                 history_report hist
#             WHERE
#                 hist.date::timestamp >= starting
#             AND
#                 hist.date::timestamp < ending
#             AND
#                 hist."totalProduction" > production_threshold
#             ORDER BY
#                 hist."homeownerId",
#                 hist.date
#         ;END;$$;
#         """)
#     pg.run()

#     # Create postgres function to get neighbors
#     pg.query("""
#         -------------------------------------------------------------------
#         -------------------------------------------------------------------
#         CREATE OR REPLACE FUNCTION
#             bodi_get_proximal_homeowners(
#                 min_miles double precision,
#                 max_miles double precision,
#                 min_neighbors integer,
#                 max_neighbors integer
#             )
#         RETURNS
#             table (

#                 homeowner_id1 integer,
#                 homeowner_id2 integer,
#                 distance_miles double precision
#             )
#         LANGUAGE plpgsql
#         AS $$
#         BEGIN
#         RETURN QUERY
#             --------------------------------------------------------------------------------------
#             -- Grab a table of all active homeowners
#             --------------------------------------------------------------------------------------
#             WITH raw AS (
#                 SELECT
#                     id,
#                     (lat::NUMERIC) * PI() / 180 AS theta,
#                     (lng::NUMERIC) * PI() / 180 AS phi
#                 FROM
#                     homeowners
#                 WHERE
#                     "isDisable"=false
#             ),

#             --------------------------------------------------------------------------------------
#             -- I'll need an earth radius variable, so define it as a CTE
#             --------------------------------------------------------------------------------------
#             earth AS (SELECT 3958 as radius),

#             --------------------------------------------------------------------------------------
#             -- This is a cross join of all homes against themselves
#             -- In which I compute the distance between each pair of homes
#             --------------------------------------------------------------------------------------
#             all_distances AS (
#                 SELECT
#                     source.id AS id1,
#                     dest.id AS id2,
#                     sqrt(
#                         ((source.theta - dest.theta) * earth.radius)^2 +
#                         (cos(source.theta) * (source.phi - dest.phi))^2
#                     ) AS ds
#                 FROM
#                     raw source
#                 CROSS JOIN
#                     raw dest
#                 CROSS JOIN
#                     earth
#             ),

#             --------------------------------------------------------------------------------------
#             -- This limits the cross join to only pairs of homes that are within
#             -- a min/max distance range.  Each id1 is ordered by increasing distance
#             --------------------------------------------------------------------------------------

#             filtered_distances AS (
#                 SELECT
#                     *,
#                     row_number() OVER (PARTITION BY id1 ORDER BY ds ASC, id2) AS seq
#                 FROM
#                     all_distances
#                 WHERE
#                     ds >= min_miles
#                 AND
#                     ds < max_miles
#                 ORDER BY
#                     id1,
#                     ds
#             ),

#             --------------------------------------------------------------------------------------
#             -- This selects the N nearest neighbors for each home
#             --------------------------------------------------------------------------------------
#             closest_distances AS (
#                 SELECT
#                     *
#                 FROM
#                     filtered_distances
#                 WHERE
#                     seq <= max_neighbors
#             ),
#             --------------------------------------------------------------------------------------
#             -- This gets a list of ids from homes that have at least some lower threshold of valid
#             -- neighbors
#             --------------------------------------------------------------------------------------
#             valid_ids AS (
#                 SELECT
#                     id1
#                 FROM
#                     closest_distances
#                 GROUP BY
#                     id1
#                 HAVING
#                     count(*) >= min_neighbors
#             )

#             --------------------------------------------------------------------------------------
#             -- This returns the closest N pairs of homes that have at leas M valid neighbors.
#             --------------------------------------------------------------------------------------
#             SELECT
#                 closest_distances.id1 AS homeowner_id1,
#                 closest_distances.id2 AS homeowner_id2,
#                 closest_distances.ds as distance_miles
#             FROM
#                 closest_distances
#             JOIN
#                 valid_ids
#             ON
#                 closest_distances.id1 = valid_ids.id1

#         ;END;$$;
#         """)
#     pg.run()
