import os
import tqdm

# 该脚本的作用是将folder_path_A标签文件夹里含有category_lists_saitisan中的[0-18]改成10-18，
#  category_lists_saitisan_fujia中的[0-4]改成20-23

category_lists_saitisan = [['Nimitz_Aircraft_Carrier', 'Kitty_Hawk_class_aircraft_carrier', '001-aircraft_carrier', 'Forrestal-class_Aircraft_Carrier'],
                           ['Wasp-class_amphibious_assault_ship', 'Tarawa-class_amphibious_assault_ship',
                            'Izumo-class_helicopter_destroyer', 'Hyuga-class_helicopter_destroyer'],
                           ['Oliver_Hazard_Perry-class_frigate', '053H1G-frigate', '053H2G-frigate', '053H3-frigate',
                            '051B-destroyer', '055-destroyer', 'Takanami-class_destroyer', 'Zumwalt-class_destroyer',
                            'Abukuma-class_destroyer_escort', 'Murasame-calss_destroyer', 'Arleigh_Burke-class_Destroyer',
                            'Kongo-class_destroyer', 'Akizuki-class_destroyer', 'Hatsuyuki-class_destroyer', 'Sovremenny-class_destroyer',
                            'Hatakaze-class_destroyer', '052-destroyer', 'Ticonderoga-class_cruiser', '054-frigate',
                            '054A-frigate', '051C-destroyer', '052B-destroyer', '052C-destroyer', '052D-destroyer', '051-destroyer',
                            '056-corvette', 'Asagiri-class_Destroyer',],
                           ['022-missile_boat', '037II-missile_boat', 'Hayabusa-class_guided-missile_patrol_boats',
                            'Cyclone-class_patrol_ship'],
                           ['529-Minesweeper', 'Hatsushima-class_minesweeper', 'Yaeyama-class_minesweeper', 'Uraga-class_Minesweeper_Tender',
                            'Uwajima-class_minesweepers', '081-Minesweeper', 'Sugashima-class_minesweepers', '082II-Minesweeper',],
                           ['072A-landing_ship', '073-landing_ship', '072III-landing_ship', '072II-landing_ship', 'JMSDF_LCU-2001_class_utility_landing_crafts',
                            'Osumi-class_landing_ship', 'Whidbey_Island-class_dock_landing_ship', '074A-landing_ship', '074-landing_ship'],
                           ['Mashu-class_replenishment_oilers', '908-replenishment_ship', '903-replenishment_ship', '903A-replenishment_ship',
                            '905-replenishment_ship', 'Towada-class_replenishment_oilers', 'Henry_J._Kaiser-class_replenishment_oiler'],
                           ['901-fast_combat_support_ship', 'Hiuchi-class_auxiliary_multi-purpose_support_ship', 'Sacramento-class_fast_combat_support_ship'],
                           ['Independence-class_littoral_combat_ship', 'Freedom-class_littoral_combat_ship']
                            ]
category_lists_saitisan_fujia = [['Iowa-class_battle_ship'],
                                 ['San_Antonio-class_amphibious_transport_dock', '071-amohibious_transport_dock'],
                                 ['Submarine', '037-submarine_chaser'],
                                 ['Other_Warship']]


def process_labels_in_folder(out_folder_path, folder_path, category_lists_saitisan, category_lists_saitisan_fujia):
    for filename in tqdm.tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            with open(os.path.join(out_folder_path, filename), 'w') as file:
                for line in lines:
                    line = line.strip()
                    data = line.split(' ')
                    class_name = data[-2]
                    class_mapped = False

                    # Check if the class is in the main category list
                    for idx, category_list in enumerate(category_lists_saitisan):
                        if class_name in category_list:
                            data[-2] = str(idx + 10)  # Map to 10, 11, 12, ...
                            new_line = ' '.join(data)
                            file.write(new_line + '\n')
                            class_mapped = True
                            break

                    if not class_mapped:
                        # Check if the class is in the additional category list
                        for idx, category_list in enumerate(category_lists_saitisan_fujia):
                            if class_name in category_list:
                                data[-2] = str(idx + 20)  # Map to 20, 21, 22, ...
                                new_line = ' '.join(data)
                                file.write(new_line + '\n')
                                class_mapped = True
                                break


# Process the labels in the folder
folder_path_A = r"I:\match\match3\train_data\labelTxt_origina_1"
out_folder_path = r'I:\match\match3\train_data\labelTxt_out_1'
process_labels_in_folder(out_folder_path, folder_path_A, category_lists_saitisan, category_lists_saitisan_fujia)
