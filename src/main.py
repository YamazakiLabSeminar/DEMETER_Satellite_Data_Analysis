from paths import PROJECT_ROOT, RAW_ROOT, RAW_DIR, OUT_ROOT, INTERIM_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR, ensure_dirs

def main():
    ensure_dirs()
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("RAW_ROOT:", RAW_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("RAW_DIR:", RAW_DIR)
    print("INTERIM_DIR:", INTERIM_DIR)
    print("TABLES_DIR:", TABLES_DIR)
    print("FIGURES_DIR:", FIGURES_DIR)
    print("LOGS_DIR:", LOGS_DIR)

if __name__ == "__main__":
    main()