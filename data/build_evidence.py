import sqlite3
import ijson
import os

def build_evidence_db(json_path, db_path):
    if os.path.exists(db_path):
        print(f"Database {db_path} already exists. Skipping build.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, text TEXT)')
    
    print("Building database... This may take a few minutes.")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        items = ijson.kvitems(f, '')
        
        batch = []
        for count, (ev_id, ev_text) in enumerate(items):
            batch.append((ev_id, ev_text))
            
            if len(batch) >= 1000:
                cursor.executemany('INSERT OR IGNORE INTO evidence (id, text) VALUES (?, ?)', batch)
                batch = []
                if count % 10000 == 0:
                    print(f"Processed {count} items...")

        if batch:
            cursor.executemany('INSERT OR IGNORE INTO evidence (id, text) VALUES (?, ?)', batch)

    conn.commit()
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_id ON evidence (id)')
    conn.close()
    print("Database build complete!")

build_evidence_db('evidence.json', 'evidence.db')