import pandas as pd
import duckdb
import glob
import re

def read_supplementary_data():
    supplementary_data = pd.read_csv('data/supplementary_data.csv')
    return supplementary_data

def read_sumer_coverages_player_play():
    return pd.read_parquet('data/sumer_coverages_player_play.parquet')

if __name__ == "__main__":
    supplementary_data = read_supplementary_data()
    sumer_coverages_player_play = read_sumer_coverages_player_play()
    
    sumer_has_targeted_defender = sumer_coverages_player_play[sumer_coverages_player_play['targeted_defender'] == True]

    sumer_useable_plays = sumer_has_targeted_defender[['game_id', 'play_id']].drop_duplicates()
    # Also get targeted defender nfl_ids for later use
    sumer_targeted_defenders = sumer_has_targeted_defender[['game_id', 'play_id', 'nfl_id']].drop_duplicates()

    underneath_routes = ['HITCH', 'OUT', 'SLANT', 'IN', 'FLAT', 'CROSS', 'ANGLE']
    underneath_plays = supplementary_data[supplementary_data['route_of_targeted_receiver'].isin(underneath_routes)]

    usable_underneath_plays = underneath_plays[['game_id', 'play_id']].drop_duplicates()
    
    # Find intersection with all usable plays
    combined_usable_plays = pd.merge(
        sumer_useable_plays, 
        usable_underneath_plays, 
        on=['game_id', 'play_id'], 
        how='inner'
    )

    print(combined_usable_plays.head())
    print('Number of usable plays: ', len(combined_usable_plays))
    
    # Read all output CSV files and combine them
    conn = duckdb.connect()
    
    conn.register('usable_plays', combined_usable_plays)
    conn.register('targeted_defenders', sumer_targeted_defenders)
    
    # Get all input CSV files
    input_files = sorted(glob.glob('data/train/input*.csv'))
    
    if not input_files:
        raise ValueError("No input*.csv files found in data/train/")
    
    print(f"Found {len(input_files)} input files to process for targeted receivers, ball landing coordinates, and play constants")
    
    # Read all input CSV files to get targeted receiver nfl_ids, ball landing coordinates, and play constants
    input_queries = []
    for file in input_files:
        input_queries.append(f"""
            SELECT DISTINCT
                CAST(game_id AS BIGINT) AS game_id,
                CAST(play_id AS BIGINT) AS play_id,
                CAST(nfl_id AS BIGINT) AS nfl_id,
                CAST(ball_land_x AS DOUBLE) AS ball_land_x,
                CAST(ball_land_y AS DOUBLE) AS ball_land_y,
                CAST(absolute_yardline_number AS DOUBLE) AS absolute_yardline_number,
                play_direction
            FROM read_csv_auto('{file}', all_varchar=True, quote='\"', strict_mode=false)
            WHERE player_role = 'Targeted Receiver'
                AND ball_land_x IS NOT NULL 
                AND ball_land_y IS NOT NULL
        """)
    
    combined_input_query = " UNION ALL ".join(input_queries)
    
    # CTE for targeted receivers with ball landing coordinates and play constants
    targeted_receivers_cte = f"""
        targeted_receivers AS (
            SELECT DISTINCT
                game_id,
                play_id,
                nfl_id,
                ball_land_x,
                ball_land_y,
                absolute_yardline_number,
                play_direction
            FROM (
                {combined_input_query}
            )
            WHERE (game_id, play_id) IN (SELECT game_id, play_id FROM usable_plays)
        ),
        targeted_defenders_cte AS (
            SELECT DISTINCT
                game_id,
                play_id,
                nfl_id
            FROM targeted_defenders
            WHERE (game_id, play_id) IN (SELECT game_id, play_id FROM usable_plays)
        ),
        ball_landing_coords AS (
            SELECT DISTINCT
                game_id,
                play_id,
                ball_land_x,
                ball_land_y
            FROM targeted_receivers
        ),
        play_constants AS (
            SELECT DISTINCT
                game_id,
                play_id,
                absolute_yardline_number,
                play_direction
            FROM targeted_receivers
        )"""
    
    # Get all output CSV files
    output_files = sorted(glob.glob('data/train/output_2023_w*.csv'))
    
    if not output_files:
        raise ValueError("No output_2023_w*.csv files found in data/train/")
    
    print(f"Found {len(output_files)} output files to process")
    
    columns_to_select = [
        'game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y'
    ]
    
    # Type casting
    cast_mappings = {
        'game_id': 'BIGINT',
        'play_id': 'BIGINT',
        'nfl_id': 'BIGINT',
        'frame_id': 'BIGINT',
        'x': 'DOUBLE',
        'y': 'DOUBLE'
    }
    columns_str = ', '.join([f"CAST(csv.{col} AS {cast_mappings[col]}) AS {col}" for col in columns_to_select])
    
    # UNION ALL query to read all output files
    output_queries = []
    for file_path in output_files:
        week_match = re.search(r'w(\d+)\.csv$', file_path)
        if week_match:
            week = int(week_match.group(1))
            output_queries.append(f"""
                SELECT {columns_str}, {week} as week
                FROM read_csv('{file_path}', all_varchar=true, quote='"', strict_mode=false) csv
            """)
    
    if output_queries:
        combined_output_query = " UNION ALL ".join(output_queries)
        
        # Final query: filter output data for both targeted receivers and defenders
        # and join with ball landing coordinates, add player_role column
        final_query = f"""
        WITH {targeted_receivers_cte},
        all_output_data AS (
            {combined_output_query}
        ),
        receiver_output AS (
            SELECT 
                output.game_id,
                output.play_id,
                output.nfl_id,
                output.frame_id,
                output.x,
                output.y,
                output.week,
                blc.ball_land_x,
                blc.ball_land_y,
                pc.absolute_yardline_number,
                pc.play_direction,
                'Targeted Receiver' AS player_role
            FROM all_output_data AS output
            INNER JOIN targeted_receivers AS tr
            ON output.game_id = tr.game_id
                AND output.play_id = tr.play_id
                AND output.nfl_id = tr.nfl_id
            INNER JOIN ball_landing_coords AS blc
            ON output.game_id = blc.game_id
                AND output.play_id = blc.play_id
            INNER JOIN play_constants AS pc
            ON output.game_id = pc.game_id
                AND output.play_id = pc.play_id
        ),
        defender_output AS (
            SELECT 
                output.game_id,
                output.play_id,
                output.nfl_id,
                output.frame_id,
                output.x,
                output.y,
                output.week,
                blc.ball_land_x,
                blc.ball_land_y,
                pc.absolute_yardline_number,
                pc.play_direction,
                'Defensive Coverage' AS player_role
            FROM all_output_data AS output
            INNER JOIN targeted_defenders_cte AS td
            ON output.game_id = td.game_id
                AND output.play_id = td.play_id
                AND output.nfl_id = td.nfl_id
            INNER JOIN ball_landing_coords AS blc
            ON output.game_id = blc.game_id
                AND output.play_id = blc.play_id
            INNER JOIN play_constants AS pc
            ON output.game_id = pc.game_id
                AND output.play_id = pc.play_id
        )
        SELECT * FROM receiver_output
        UNION ALL
        SELECT * FROM defender_output
        """
        
        # Execute query and save to parquet
        result_df = conn.execute(final_query).df()
        output_path = 'data/reachability_label_data.parquet'
        result_df.to_parquet(output_path, index=False)
        
        print(f'\nCombined data saved to {output_path}')
        print(f'Total rows: {len(result_df)}')
    
    conn.close()
