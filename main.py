
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

SCRIPTS = {
    "1": {
        "name": "Check Original Dataset",
        "file": "check_dataset.py",
    },
    "2": {
        "name": "Preprocess Sample Images",
        "file": "preprocess.py",
    },
    "3": {
        "name": "Split Dataset",
        "file": "split_dataset.py",
    },
    "4": {
        "name": "Train CNN Model",
        "file": "train.py",
    },
    "5": {
        "name": "Evaluate CNN Model",
        "file": "evaluate.py",
    },
    "6": {
        "name": "Predict Test Images",
        "file": "predict.py",
    },
}


# ==========================================
# 2. DISPLAY MENU
# ==========================================

def display_menu():
    print("\n" + "=" * 55)
    print("       APPLE LEAF CLASSIFICATION SYSTEM")
    print("=" * 55)

    for option, details in SCRIPTS.items():
        print(f"  {option}. {details['name']}")

    print("  0. Exit")

    print("=" * 55)


# ==========================================
# 3. CONFIRM DESTRUCTIVE OR LONG TASKS
# ==========================================

def confirm_action(message):
    answer = input(f"\n{message} (y/n): ").strip().lower()
    return answer == "y"


# ==========================================
# 4. RUN AN EXISTING PYTHON SCRIPT
# ==========================================


def run_script(script_name):
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        print(f"\n[ERROR] Script not found: {script_path}")
        return

    print("\n" + "=" * 55)
    print(f" OPERATION : {script_name}")
    print(f" STARTED   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=False,
        )

        elapsed_time = time.perf_counter() - start_time

        print("\n" + "=" * 55)

        if result.returncode == 0:
            print("[SUCCESS] Operation completed.")
        else:
            print("[FAILED] Operation returned an error.")

        print(f" SCRIPT    : {script_name}")
        print(f" EXIT CODE : {result.returncode}")
        print(f" DURATION  : {elapsed_time:.2f} seconds")
        print(f" FINISHED  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 55)

    except KeyboardInterrupt:
        elapsed_time = time.perf_counter() - start_time
        print("\n[INTERRUPTED] Operation stopped by user.")
        print(f"Duration before interruption: {elapsed_time:.2f} seconds")

    except OSError as error:
        print(f"\n[ERROR] Could not start script: {error}")



# ==========================================
# 5. HANDLE MENU SELECTION
# ==========================================

def handle_selection(choice):
    if choice == "0":
        print("\nThank you for using the classification system!")
        return False

    if choice not in SCRIPTS:
        print("\n[WARNING] Invalid option. Try again.")
        return True

    selected = SCRIPTS[choice]
    script_name = selected["file"]

    # Dataset splitting must not overwrite existing data.
    if choice == "3":
        split_dir = PROJECT_ROOT / "dataset" / "split"

        if split_dir.exists() and any(
            path.is_file() for path in split_dir.rglob("*")):
            print("\n[WARNING] A split dataset already exists.")
            print(f"Location: {split_dir}")
            print(
                "The splitting script protects existing data "
                "and will not overwrite it."
            )
            return True

        if not confirm_action(
            "This will create train, validation, and test folders. Continue?"
        ):
            print("Dataset splitting cancelled.")
            return True

    # Training can take time and overwrites saved model files.
    elif choice == "4":
        if not confirm_action(
            "Training may take several minutes and update saved models. Continue?"
        ):
            print("Training cancelled.")
            return True

    # Prediction deletes the previous prediction-results folder.
    elif choice == "6":
        if not confirm_action(
            "This will delete and recreate results/predictions. Continue?"
        ):
            print("Prediction cancelled.")
            return True

    run_script(script_name)
    return True


# ==========================================
# 6. MAIN APPLICATION LOOP
# ==========================================

def main():
    print("\nWelcome to the Apple Leaf CNN Project!")

    while True:
        display_menu()

        choice = input("\nSelect an option: ").strip()

        should_continue = handle_selection(choice)

        if not should_continue:
            break

        input("\nPress Enter to return to the menu...")


# ==========================================
# 7. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
