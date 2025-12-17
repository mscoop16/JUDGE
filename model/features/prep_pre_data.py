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

    sumer_useable_plays = sumer_has_targeted_defender[['game_id', 'play_id', 'nfl_id']].drop_duplicates()

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
    
    # Read all input CSV files and combine them
    conn = duckdb.connect()
    
    conn.register('usable_plays', combined_usable_plays)
    
    csv_files = sorted(glob.glob('data/train/input_2023_w*.csv'))
    
    columns_to_select = [
        'game_id', 'play_id', 'player_to_predict', 'nfl_id', 'frame_id',
        'play_direction', 'absolute_yardline_number', 'player_name',
        'player_position', 'player_side', 'player_role',
        'x', 'y', 's', 'a', 'dir', 'o',
        'num_frames_output', 'ball_land_x', 'ball_land_y'
    ]
    
    # Type casting
    cast_mappings = {
        'game_id': 'BIGINT',
        'play_id': 'BIGINT',
        'player_to_predict': 'BOOLEAN',
        'nfl_id': 'BIGINT',
        'frame_id': 'BIGINT',
        'play_direction': 'VARCHAR',
        'absolute_yardline_number': 'DOUBLE',
        'player_name': 'VARCHAR',
        'player_position': 'VARCHAR',
        'player_side': 'VARCHAR',
        'player_role': 'VARCHAR',
        'x': 'DOUBLE',
        'y': 'DOUBLE',
        's': 'DOUBLE',
        'a': 'DOUBLE',
        'dir': 'DOUBLE',
        'o': 'DOUBLE',
        'num_frames_output': 'BIGINT',
        'ball_land_x': 'DOUBLE',
        'ball_land_y': 'DOUBLE'
    }
    columns_str = ', '.join([f"CAST(csv.{col} AS {cast_mappings[col]}) AS {col}" for col in columns_to_select])
    
    # UNION ALL query
    queries = []
    for file_path in csv_files:
        week_match = re.search(r'w(\d+)\.csv$', file_path)
        if week_match:
            week = int(week_match.group(1))
            queries.append(f"""
                SELECT {columns_str}, {week} as week
                FROM read_csv('{file_path}', all_varchar=true, quote='"', strict_mode=false) csv
                INNER JOIN usable_plays up
                ON CAST(csv.game_id AS BIGINT) = up.game_id 
                AND CAST(csv.play_id AS BIGINT) = up.play_id 
                AND CAST(csv.nfl_id AS BIGINT) = up.nfl_id
            """)
    
    if queries:
        # Combine all queries with UNION ALL
        combined_query = " UNION ALL ".join(queries)
        result_df = conn.execute(combined_query).df()
        
        # Save to parquet
        output_path = 'data/underneath_input_tracking_data.parquet'
        result_df.to_parquet(output_path, index=False)
        
        print(f'\nCombined data saved to {output_path}')
        print(f'Total rows: {len(result_df)}')
        print(f'Week range: {result_df["week"].min()} - {result_df["week"].max()}')
    
    conn.close()