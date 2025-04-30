import sqlite3


def init_db(path="cavities.db"):
    conn = sqlite3.connect(path)
    c = conn.cursor()

    # Author table
    c.execute('''
        CREATE TABLE IF NOT EXISTS authors (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        )
    ''')

    # SHG table
    c.execute('''
        CREATE TABLE IF NOT EXISTS shg (
            id INTEGER PRIMARY KEY,
            author_id INTEGER,
            cavity_length_mm REAL,
            crystal_length_mm REAL,
            waist_w0_um REAL,
            waist_w1_um REAL,
            folding_angle_deg REAL,
            roc1_mm REAL,
            roc2_mm REAL,
            input_wavelength_nm REAL,
            output_wavelength_nm REAL,
            input_power_mW REAL,
            output_power_mW REAL,
            T_input_coupler REAL,
            FOREIGN KEY(author_id) REFERENCES authors(id)
        )
    ''')

    # OPO table
    c.execute('''
        CREATE TABLE IF NOT EXISTS opo (
            id INTEGER PRIMARY KEY,
            author_id INTEGER,
            cavity_length_mm REAL,
            crystal_length_mm REAL,
            waist_w0_um REAL,
            waist_w1_um REAL,
            folding_angle_deg REAL,
            roc1_mm REAL,
            roc2_mm REAL,
            input_wavelength_nm REAL,
            output_wavelength_nm REAL,
            input_power_mW REAL,
            output_power_mW REAL,
            T_output_coupler REAL,
            threshold_power_mW REAL, 
            squeezing_dB REAL,
            FOREIGN KEY(author_id) REFERENCES authors(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("SQLite DB initialized.")


if __name__ == "__main__":
    init_db()
