import sqlite3

DB_PATH = "cavities.db"


def get_author_id(conn, name):
    cur = conn.cursor()
    cur.execute("SELECT id FROM authors WHERE name = ?", (name,))
    result = cur.fetchone()
    if result:
        return result[0]
    # Else, insert author
    cur.execute("INSERT INTO authors (name) VALUES (?)", (name,))
    return cur.lastrowid


def add_entry(author_name, shg_data=None, opo_data=None):
    conn = sqlite3.connect(DB_PATH)
    author_id = get_author_id(conn, author_name)
    cur = conn.cursor()

    if shg_data:
        fields = ", ".join(shg_data.keys())
        placeholders = ", ".join(["?"] * len(shg_data))
        values = list(shg_data.values())
        cur.execute(f"INSERT INTO shg (author_id, {fields}) VALUES (?, {placeholders})", [author_id] + values)

    if opo_data:
        fields = ", ".join(opo_data.keys())
        placeholders = ", ".join(["?"] * len(opo_data))
        values = list(opo_data.values())
        cur.execute(f"INSERT INTO opo (author_id, {fields}) VALUES (?, {placeholders})", [author_id] + values)

    conn.commit()
    conn.close()
    print(f"Data added for {author_name}")


def modify_entry(table, record_id, field, new_value):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        cur.execute(f"UPDATE {table} SET {field} = ? WHERE id = ?", (new_value, record_id))
        conn.commit()
        print(f"{table} ID {record_id}: {field} updated to {new_value}")
    except sqlite3.OperationalError as e:
        print("Error:", e)
    finally:
        conn.close()


def add_column(table, column_name, column_type="REAL"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")
        print(f"Added column '{column_name}' to '{table}'")
        conn.commit()
    except sqlite3.OperationalError as e:
        print("Error:", e)
    finally:
        conn.close()


def load_all_data():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # So we can get dicts
    cur = conn.cursor()

    data = {}

    # Get all authors
    cur.execute("SELECT * FROM authors")
    authors = cur.fetchall()

    for author in authors:
        author_name = author["name"]
        author_id = author["id"]
        data[author_name] = {"shg": [], "opo": []}

        # Fetch SHG entries
        cur.execute("SELECT * FROM shg WHERE author_id = ?", (author_id,))
        shg_entries = cur.fetchall()
        data[author_name]["shg"] = [dict(row) for row in shg_entries]

        # Fetch OPO entries
        cur.execute("SELECT * FROM opo WHERE author_id = ?", (author_id,))
        opo_entries = cur.fetchall()
        data[author_name]["opo"] = [dict(row) for row in opo_entries]

    conn.close()
    return data