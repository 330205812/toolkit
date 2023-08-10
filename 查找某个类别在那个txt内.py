import os

def find_files_with_class(folder_path, target_class):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    data = line.split(' ')
                    if data[-2] == target_class:  # Check if the class matches the target class
                        print(f"Found '{target_class}' in file: {filename}")
                        break

if __name__ == "__main__":
    folder_path_A = r"I:\match\match3\train_data\labelTxt_new"
    target_class = '037-submarine_chaser'
    find_files_with_class(folder_path_A, target_class)
