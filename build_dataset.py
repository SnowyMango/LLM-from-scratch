import os
import sysconfig

OUTPUT = "input.txt"
NOTES_DIR = "notes"
NOTES_REPEAT = 2       # repeat notes so they carry more weight than the code
MAX_MB = 5             # total size cap for input.txt
SKIP_DIRS = {"test", "tests", "__pycache__", "site-packages", "idlelib", "lib2to3"}


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def collect_notes():
    notes = []
    if not os.path.isdir(NOTES_DIR):
        return notes
    for dirpath, dirnames, filenames in os.walk(NOTES_DIR):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith((".txt", ".md")):
                path = os.path.join(dirpath, name)
                text = read_text(path)
                if text:
                    notes.append((os.path.relpath(path, NOTES_DIR), text))
    return notes


def collect_python_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith("test"))
        for name in sorted(filenames):
            if name.endswith(".py") and not name.startswith("test_"):
                yield os.path.join(dirpath, name)


def main():
    budget = MAX_MB * 1024 * 1024
    total = 0
    parts = []

    notes = collect_notes()
    for rel, text in notes:
        block = f"\n# ===== note: {rel} =====\n{text}\n"
        for _ in range(NOTES_REPEAT):
            parts.append(block)
            total += len(block.encode("utf-8"))
    print(f"study notes: {len(notes)} file(s), each repeated {NOTES_REPEAT} times")

    stdlib = sysconfig.get_paths()["stdlib"]
    print(f"reading Python source from: {stdlib}")
    file_count = 0
    for path in collect_python_files(stdlib):
        if total >= budget:
            break
        text = read_text(path)
        if not text:
            continue
        rel = os.path.relpath(path, stdlib)
        part = f"\n# ===== {rel} =====\n{text}\n"
        parts.append(part)
        total += len(part.encode("utf-8"))
        file_count += 1

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("".join(parts))

    print(f"python files included: {file_count}")
    print(f"wrote {OUTPUT} ({total / 1024 / 1024:.2f} MB)")
    print("If you already have a mini_llm.pt, delete it before retraining on the new data.")


if __name__ == "__main__":
    main()
