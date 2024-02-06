import numpy as np
import os
import evaluate_util as e_util
import re

def generate_latex_tables():

    generate_latex_tables_with_rivals()
    generate_latex_tables_versions()
    generate_latex_tables_compare_alphas()

def generate_latex_tables_with_rivals():
    
    rivals_results_path = "../segmentationResult/rivals/evaluationResult/results.npy"
    our_results_path = "../segmentationResult/ours/evaluationResult/results.npy"
    dataset_name_list = ["overpass","pedestrians","canoe", "port_0_17fps", "boats", "fountain01", "fountain02"]
    ALPHA_list = [0.001, 0.005, 0.01, 0.05]
    version_list = ["64", "32", "128", "16"]
    
    noise_level_list = ["0", "0.1", "0.2", "0.31", "mask", "mask2x2", "mask3x3", "mask4x4", "saltpepper", "uniform", "compression10", "compression5", "compression1"]
    compare_methods = ["zivkovic", "kde", "wren", "sobs", "fsom"]
    
    if (os.path.exists(rivals_results_path)):
        print("Loading rivals results")
        rivals_results = np.load(rivals_results_path)
    else:
        print("Evaluating rivals to get results.")
        rivals_results = e_util.get_evaluation_rivals()
        rivals_results = rivals_results[0]
    if (os.path.exists(our_results_path)):
        print("Loading our results")
        our_results = np.load(our_results_path)
    else:
        print("Evaluating ours to get results.")
        our_results = e_util.get_evaluation_all_ours()
        our_results = our_results[0]
    
    print(our_results.shape)
    print(rivals_results.shape)
    
    for noise_index in range(len(noise_level_list)):
        noise = noise_level_list[noise_index]
        noise = noise.replace(".", "_")
            
        #F_list = (1+len(compare_methods))*[[]]
        F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
        for dataset_name_index in range(len(dataset_name_list)):
            for version_index in range(len(version_list))[3:]:
                value = np.max(our_results[version_index, dataset_name_index, noise_index, :, 6])
                std_index = np.where(our_results[version_index, dataset_name_index, noise_index, :, 6] == value)
                std_index = std_index[0]
                std = our_results[version_index, dataset_name_index, noise_index, std_index, 14]
                F_list[dataset_name_index].append("$" + str(format(round(value, 3), '.3f')) + " \pm " + str(format(round(std, 2), '.2f')) + "$")
                                
            for compare_method_index in range(len(compare_methods)):  
                value = rivals_results[compare_method_index, dataset_name_index, noise_index, 6]
                std = rivals_results[compare_method_index, dataset_name_index, noise_index, 14]
                F_list[dataset_name_index].append("$" + str(format(round(value, 3), '.3f')) + " \pm " + str(format(round(std, 2), '.2f')) + "$")
        
        #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
        generate_latex_table([[""] + ["ours"] + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
        
def generate_latex_tables_versions():
    
    rivals_results_path = "../segmentationResult/rivals/evaluationResult/results.npy"
    our_results_path = "../segmentationResult/ours/evaluationResult/results.npy"
    dataset_name_list = ["overpass","pedestrians","canoe", "port_0_17fps", "boats", "fountain01", "fountain02"]
    ALPHA_list = [0.001, 0.005, 0.01, 0.05]
    version_list = ["64", "32", "128", "16"]
    version_list = ["16", "32", "64", "128"]
    
    noise_level_list = ["0", "0.1", "0.2", "0.31", "mask", "mask2x2", "mask3x3", "mask4x4", "saltpepper", "uniform", "compression10", "compression5", "compression1"]
    compare_methods = ["zivkovic", "kde", "wren", "sobs", "fsom"]
    
    if (os.path.exists(our_results_path)):
        print("Loading our results")
        our_results = np.load(our_results_path)
    else:
        print("Evaluating ours to get results.")
        our_results = e_util.get_evaluation_all_ours()
    
    print(our_results.shape)
    
            
    #F_list = (1+len(compare_methods))*[[]]
    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for version_index in [3,1,0,2]:
            value = np.mean(our_results[version_index, dataset_name_index, 0, :, 6], axis = 0)
            std = np.mean(our_results[version_index, dataset_name_index, 0, :, 14], axis = 0)
            F_list[dataset_name_index].append("$" + str(format(round(value, 3), '.3f')) + " \pm " + str(format(round(std, 2), '.2f')) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + version_list] + F_list, noise_level_list[0], noise_level_list[0],  "test.txt")

    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for version_index in [3,1,0,2]:
            value = np.mean(np.mean(our_results[version_index, dataset_name_index, 1:4, :, 6], axis = 0), axis = 0)
            std = np.mean(np.mean(our_results[version_index, dataset_name_index, 1:4, :, 14], axis = 0), axis = 0)
            F_list[dataset_name_index].append("$" + str(format(round(value, 3), '.3f')) + " \pm " +  str(format(round(std, 2), '.2f')) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + version_list] + F_list, "Gaussian", "Gaussian",  "test.txt")
    
    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for version_index in [3,1,0,2]:
            value = np.mean(np.mean(our_results[version_index, dataset_name_index, 4:8, :, 6], axis = 0), axis = 0)
            std = np.mean(np.mean(our_results[version_index, dataset_name_index, 4:8, :, 14], axis = 0), axis = 0)
            F_list[dataset_name_index].append("$" + str(format(round(value, 3), '.3f')) + " \pm " + str(format(round(std, 2), '.2f')) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + version_list] + F_list, "Mask", "Mask",  "test.txt")
    
    #F_list = (1+len(compare_methods))*[[]]
    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for version_index in [3,1,0,2]:
            value = np.mean(our_results[version_index, dataset_name_index, 8, :, 6], axis = 0)
            std = np.mean(our_results[version_index, dataset_name_index, 8, :, 14], axis = 0)
            F_list[dataset_name_index].append("$" + str(format(round(value, 3), '.3f')) + " \pm " + str(format(round(std, 2), '.2f')) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + version_list] + F_list, "saltpepper", "saltpepper",  "test.txt")
    
    #F_list = (1+len(compare_methods))*[[]]
    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for version_index in [3,1,0,2]:
            value = np.mean(our_results[version_index, dataset_name_index, 9, :, 6], axis = 0)
            std = np.mean(our_results[version_index, dataset_name_index, 9, :, 14], axis = 0)
            F_list[dataset_name_index].append("$" + str(format(round(value, 3), '.3f')) + " \pm " + str(format(round(std, 2), '.2f')) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + version_list] + F_list, "uniform", "uniform",  "test.txt")
    
    #F_list = (1+len(compare_methods))*[[]]
    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for version_index in [3,1,0,2]:
            value = np.mean(np.mean(our_results[version_index, dataset_name_index, 10:12, :, 6], axis = 0), axis = 0)
            std = np.mean(np.mean(our_results[version_index, dataset_name_index, 10:12, :, 14], axis = 0), axis = 0)
            F_list[dataset_name_index].append("$" + str(format(round(value, 3), '.3f')) + " \pm " +   str(format(round(std, 2), '.2f')) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + version_list] + F_list, "compression", "compression",  "test.txt")
    
def generate_latex_tables_compare_alphas():
    
    rivals_results_path = "../segmentationResult/rivals/evaluationResult/results.npy"
    our_results_path = "../segmentationResult/ours/evaluationResult/results.npy"
    dataset_name_list = ["overpass","pedestrians","canoe", "port_0_17fps", "boats", "fountain01", "fountain02"]
    ALPHA_list = [0.001, 0.005, 0.01, 0.05]
    version_list = ["64", "32", "128", "16"]
    
    noise_level_list = ["0", "0.1", "0.2", "0.31", "mask", "mask2x2", "mask3x3", "mask4x4", "saltpepper", "uniform", "compression"]
    compare_methods = ["zivkovic", "kde", "wren", "sobs", "fsom"]
    
    if (os.path.exists(our_results_path)):
        print("Loading our results")
        our_results = np.load(our_results_path)
    else:
        print("Evaluating ours to get results.")
        our_results = e_util.get_evaluation_all_ours()
    
    print(our_results.shape)
    
            
    #F_list = (1+len(compare_methods))*[[]]
    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for ALPHA in range(len(ALPHA_list)):
            F_list[dataset_name_index].append("$" + str(round(our_results[3, dataset_name_index, 0, ALPHA, 6], 3)) + " \pm " +  str(round(our_results[3, dataset_name_index, 0, ALPHA, 14], 2)) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + ALPHA_list] + F_list, noise_level_list[0], noise_level_list[0],  "test.txt")

    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for ALPHA in range(len(ALPHA_list)):
            F_list[dataset_name_index].append("$" + str(round(np.mean(our_results[3, dataset_name_index, 1:4, ALPHA, 6], axis=0), 3)) + " \pm " +  str(round(np.mean(our_results[3, dataset_name_index, 1:4, ALPHA, 14], axis = 0), 2)) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + ALPHA_list] + F_list, "Gaussian", "Gaussian",  "test.txt")
    
    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for ALPHA in range(len(ALPHA_list)):
            F_list[dataset_name_index].append("$" + str(round(np.mean(our_results[3, dataset_name_index, 4:8, ALPHA, 6], axis=0), 3)) + " \pm " +  str(round(np.mean(our_results[3, dataset_name_index, 4:8, ALPHA, 14], axis=0), 2)) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + ALPHA_list] + F_list, "Mask", "Mask",  "test.txt")
    
    #F_list = (1+len(compare_methods))*[[]]
    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for ALPHA in range(len(ALPHA_list)):
            F_list[dataset_name_index].append("$" + str(round(our_results[3, dataset_name_index, 8, ALPHA, 6], 3)) + " \pm " +  str(round(our_results[3, dataset_name_index, 8, ALPHA, 14], 2)) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + ALPHA_list] + F_list, "saltpepper", "saltpepper",  "test.txt")
    
    #F_list = (1+len(compare_methods))*[[]]
    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for ALPHA in range(len(ALPHA_list)):
            F_list[dataset_name_index].append("$" + str(round(our_results[3, dataset_name_index, 9, ALPHA, 6], 3)) + " \pm " +  str(round(our_results[3, dataset_name_index, 9, ALPHA, 14], 2)) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + ALPHA_list] + F_list, "uniform", "uniform",  "test.txt")
    
    #F_list = (1+len(compare_methods))*[[]]
    F_list = [["overpass"],["pedestrians"],["canoe"],["port-17fps"],["boats"],["fountain01"],["fountain02"]]
    for dataset_name_index in range(len(dataset_name_list)):
        for ALPHA in range(len(ALPHA_list)):
            F_list[dataset_name_index].append("$" + str(round(our_results[3, dataset_name_index, 10, ALPHA, 6], 3)) + " \pm " +  str(round(our_results[3, dataset_name_index, 10, ALPHA, 14], 2)) + "$")
        
    #generate_latex_table([[""] + version_list + compare_methods] + F_list, noise_level_list[noise_index], noise_level_list[noise_index],  "test.txt")
    generate_latex_table([[""] + ALPHA_list] + F_list, "compression", "compression",  "test.txt")
            
def generate_latex_table(data_to_table, table_caption, table_label, file_name):
    table_str = ""

    table_str = "\\begin{table}[]\n"
    table_str = table_str + "\centering\n"
    table_str = table_str + "\caption{" + table_caption + "}\n"
    table_str = table_str + "\label{" + table_label + "}\n"
    table_str = table_str + "\\begin{tabular}{" + len(data_to_table[0])*"l" + "}"

    index_data_line = 0
    for data_line in data_to_table:
        mark_maximum = index_data_line > 0  and len([x for x in data_line if ((re.search("\d+\.(\d*)?", x)) and isinstance(float(re.search("\d+\.(\d*)?", x).group(0)),float))]) > 0
        line = generate_latex_table_line(data_line, mark_maximum_automatically = mark_maximum)        
        table_str = table_str + "\n" + line
        index_data_line += 1
    table_str = table_str[:-2] + "\n"
    table_str = table_str + "\end{tabular}\n"
    table_str = table_str + "\end{table}\n"

    with open(file_name, 'a') as f:
        f.write(table_str)   

def generate_latex_table_line (data_to_line, mark_maximum_automatically = False, mark_list = None):
    if (mark_maximum_automatically):
        filtered_data = [float(re.search("\d+\.(\d*)?", x).group(0)) for x in data_to_line if ((re.search("\d+\.(\d*)?", x)) and isinstance(float(re.search("\d+\.(\d*)?", x).group(0)),float))]
        max_index = filtered_data.index(max(filtered_data)) + 1
        line = ""
        data_index = 0
        for data in data_to_line:
            if (data_index == max_index):
                line = line + " " + "\\boldmath" + str(data) + "\\unboldmath" + " &"
                print(line)
            else:
                line = line + " " + str(data) + " &"
            data_index += 1
    else:
        if ((mark_list is None) or (mark_list == [])):
            line = ""
            for data in data_to_line:
                line = line + " " + str(data) + " &"
                
        else:
            data_index = 0
            for data in data_to_line:
                if (data_index in mark_list):
                    line = line + " " + "\\textbf{" + str(data) + "}" + " &"
                else:
                    line = line + " " + str(data) + " &"   
                data_index += 1
            
    line = line[:-1] + "\\\\"
    return line
    
