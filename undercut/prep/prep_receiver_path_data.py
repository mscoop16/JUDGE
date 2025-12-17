import pandas as pd
import duckdb
import glob
import re
import numpy as np

def read_supplementary_data():
    supplementary_data = pd.read_csv('data/supplementary_data.csv')
    return supplementary_data

def normalize_coordinates(data):
    """
    Normalize coordinates and angles for all rows in the dataset.
    Normalizes based on play direction and line of scrimmage.
    """
    FIELD_LENGTH = 120
    FIELD_WIDTH = 53.3
    
    # Get play-level constants
    play_constants = data.groupby(['game_id', 'play_id']).agg({
        'absolute_yardline_number': 'first',
        'play_direction': 'first'
    }).reset_index()
    
    # Merge play constants
    data = data.merge(play_constants, on=['game_id', 'play_id'], suffixes=('', '_play'))
    line_of_scrimmage = data['absolute_yardline_number_play']
    play_direction = data['play_direction_play']
    
    is_left_play = play_direction == 'left'
    
    # Normalize x coordinates
    x_new = np.where(is_left_play, FIELD_LENGTH - data['x'], data['x'])
    los_new = np.where(is_left_play, FIELD_LENGTH - line_of_scrimmage, line_of_scrimmage)
    x_norm = x_new - los_new
    
    # Normalize y coordinates
    y_new = np.where(is_left_play, FIELD_WIDTH - data['y'], data['y'])
    y_centered = y_new - (FIELD_WIDTH / 2)
    
    # Update the df with normalized values
    data = data.copy()
    data['x'] = x_norm.round(2)
    data['y'] = y_centered.round(2)
    
    # Drop the temporary play constants
    data = data.drop(columns=['absolute_yardline_number_play', 'play_direction_play'])
    
    return data

if __name__ == "__main__":
    supplementary_data = read_supplementary_data()
    
    underneath_routes = ['HITCH', 'OUT', 'SLANT', 'IN', 'FLAT', 'CROSS', 'ANGLE']
    underneath_plays = supplementary_data[supplementary_data['route_of_targeted_receiver'].isin(underneath_routes)]

    usable_underneath_plays = underneath_plays[['game_id', 'play_id']].drop_duplicates()
    
    print(usable_underneath_plays.head())
    print('Number of usable underneath plays: ', len(usable_underneath_plays))
    
    # Read all input and output files and combine them
    conn = duckdb.connect()
    
    conn.register('usable_plays', usable_underneath_plays)
    
    # Get all input files
    input_files = sorted(glob.glob('data/train/input_2023_w*.csv'))
    
    if not input_files:
        raise ValueError("No input_2023_w*.csv files found in data/train/")
    
    print(f"Found {len(input_files)} input files to process")
    
    # UNION ALL query to read all input files
    input_queries = []
    for file_path in input_files:
        week_match = re.search(r'w(\d+)\.csv$', file_path)
        if week_match:
            week = int(week_match.group(1))
            input_queries.append(f"""
                SELECT 
                    CAST(game_id AS BIGINT) AS game_id,
                    CAST(play_id AS BIGINT) AS play_id,
                    CAST(nfl_id AS BIGINT) AS nfl_id,
                    CAST(frame_id AS BIGINT) AS frame_id,
                    CAST(x AS DOUBLE) AS x,
                    CAST(y AS DOUBLE) AS y,
                    CAST(absolute_yardline_number AS DOUBLE) AS absolute_yardline_number,
                    play_direction,
                    {week} as week,
                    'input' as source
                FROM read_csv('{file_path}', all_varchar=true, quote='"', strict_mode=false)
                WHERE player_role = 'Targeted Receiver'
            """)
    
    # Get all output files
    output_files = sorted(glob.glob('data/train/output_2023_w*.csv'))
    
    if not output_files:
        raise ValueError("No output_2023_w*.csv files found in data/train/")
    
    print(f"Found {len(output_files)} output files to process")
    
    # UNION ALL query to read all output files
    output_queries = []
    for file_path in output_files:
        week_match = re.search(r'w(\d+)\.csv$', file_path)
        if week_match:
            week = int(week_match.group(1))
            output_queries.append(f"""
                SELECT 
                    CAST(game_id AS BIGINT) AS game_id,
                    CAST(play_id AS BIGINT) AS play_id,
                    CAST(nfl_id AS BIGINT) AS nfl_id,
                    CAST(frame_id AS BIGINT) AS frame_id,
                    CAST(x AS DOUBLE) AS x,
                    CAST(y AS DOUBLE) AS y,
                    {week} as week,
                    'output' as source
                FROM read_csv('{file_path}', all_varchar=true, quote='"', strict_mode=false)
            """)
    
    if input_queries and output_queries:
        combined_input_query = " UNION ALL ".join(input_queries)
        combined_output_query = " UNION ALL ".join(output_queries)
        
        # Get play constants from input data
        play_constants_query = f"""
            SELECT DISTINCT 
                game_id,
                play_id,
                absolute_yardline_number,
                play_direction
            FROM ({combined_input_query})
            WHERE absolute_yardline_number IS NOT NULL 
            AND play_direction IS NOT NULL
        """
        
        # Final query: combine input and output data, filter to usable underneath plays and targeted receivers
        final_query = f"""
        WITH input_data AS (
            {combined_input_query}
        ),
        output_data AS (
            {combined_output_query}
        ),
        play_constants AS (
            {play_constants_query}
        ),
        targeted_receivers AS (
            SELECT DISTINCT game_id, play_id, nfl_id
            FROM input_data
        ),
        filtered_input AS (
            SELECT input.*
            FROM input_data AS input
            INNER JOIN usable_plays AS up
            ON input.game_id = up.game_id 
            AND input.play_id = up.play_id
        ),
        filtered_output AS (
            SELECT output.*, pc.absolute_yardline_number, pc.play_direction
            FROM output_data AS output
            INNER JOIN usable_plays AS up
            ON output.game_id = up.game_id 
            AND output.play_id = up.play_id
            INNER JOIN targeted_receivers AS tr
            ON output.game_id = tr.game_id
            AND output.play_id = tr.play_id
            AND output.nfl_id = tr.nfl_id
            INNER JOIN play_constants AS pc
            ON output.game_id = pc.game_id
            AND output.play_id = pc.play_id
        )
        SELECT 
            game_id, play_id, nfl_id, frame_id, x, y, 
            absolute_yardline_number, play_direction, week, source
        FROM filtered_input
        UNION ALL
        SELECT 
            game_id, play_id, nfl_id, frame_id, x, y, 
            absolute_yardline_number, play_direction, week, source
        FROM filtered_output
        ORDER BY game_id, play_id, nfl_id, source, frame_id
        """
        
        # Execute query and get result
        result_df = conn.execute(final_query).df()
        
        # Normalize coordinates
        print("Normalizing coordinates...")
        result_df = normalize_coordinates(result_df)
        
        # Save to parquet
        output_path = 'data/receiver_path_data.parquet'
        result_df.to_parquet(output_path, index=False)
        
        print(f'\nCombined data saved to {output_path}')
        print(f'Total rows: {len(result_df)}')
        print(f'Input rows: {len(result_df[result_df["source"] == "input"])}')
        print(f'Output rows: {len(result_df[result_df["source"] == "output"])}')
        print(f'Week range: {result_df["week"].min()} - {result_df["week"].max()}')
        print(f'Columns: {list(result_df.columns)}')
    
    conn.close()

