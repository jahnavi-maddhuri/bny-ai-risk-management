import os
import pandas as pd
from databricks import sql

def load_to_databricks():
    #FIXME update file name
    files_to_load = ['gnews_data.csv', 'secondary_data.csv'] 
    #FIXME update database name
    table_name = "risk_news_feed"  # Ensure this table exists in Databricks

    # 1. READ AND COMBINE CSVs
    dfs = []
    print("--------------------------------------------------")
    print("Step 1: Reading CSV files...")
    
    for filename in files_to_load:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                dfs.append(df)
                print(f"   Loaded {filename} ({len(df)} rows)")
            except Exception as e:
                print(f"   Error reading {filename}: {e}")
        else:
            print(f"    Warning: {filename} not found. Skipping.")

    if not dfs:
        print(" No data found in CSVs. Exiting loader.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Total rows to upload: {len(combined_df)}")

    print("Step 2: Connecting to Databricks...")
    try:
        #FIXME set env
        connection = sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN")
        )
        cursor = connection.cursor()
        print("   Connected successfully.")
    except Exception as e:
        print(f"   Connection failed. Check your environment variables.\nError: {e}")
        return

    # 3. INSERT DATA
    print(f"Step 3: Inserting data into table `{table_name}`...")
    
    #FIXME check field name
    query_start = f"INSERT INTO {table_name} (id,title,link,published,source,summary,query,fetched_at) VALUES "
    values_list = []

    for index, row in combined_df.iterrows():
        # Sanitize data to prevent SQL errors (escaping single quotes)
        # We use .get() to avoid crashing if a column is missing in the CSV
        f_id = str(row.get('feed_id', '')).replace("'", "''")
        date = str(row.get('date', '')).replace("'", "''")
        src = str(row.get('source', '')).replace("'", "''")
        content = str(row.get('content', '')).replace("'", "''")
        score = row.get('risk_score', 0) # Default to 0 if missing

        # SQL Format: ('id', 'date', 'source', 'content', score)
        val_str = f"('{f_id}', '{date}', '{src}', '{content}', {score})"
        values_list.append(val_str)

    # Batch execution (Chunking is safer for very large datasets, 
    # but for daily news <5000 rows, one batch is usually fine)
    if values_list:
        try:
            full_query = query_start + ",\n".join(values_list)
            cursor.execute(full_query)
            connection.commit() # Important! Saves the changes.
            print("   ✅ Upload complete!")
        except Exception as e:
            print(f"   ❌ SQL Execution Error: {e}")
    else:
        print("   ⚠️ No valid rows to insert.")

    # 4. CLEANUP
    cursor.close()
    connection.close()
    print("--------------------------------------------------")

if __name__ == "__main__":
    load_to_databricks()