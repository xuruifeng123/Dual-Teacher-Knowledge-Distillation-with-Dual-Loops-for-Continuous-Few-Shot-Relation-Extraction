# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import subprocess
import argparse
import os
import shutil
from data_transforms import trainsforms_data_train,trainsforms_data_text
from base_function import save_loss_png,data_sampler,eval_LLM,change_sh,change_sh_text,eval_accuracy

from transformers import AutoModelForCausalLM, pipeline
from loguru import logger
import numpy as np
def main(task):
    # train_eval
    result_whole_test=[]
    j=5
    for rou in range(1):
        test_total=[]
        if task =="fewrel":
            sampler = data_sampler(config=80, seed=100 + rou * 100,
                                test_file=f"data/{task}/CFRLdata_10_100_10_5/test_0.txt",
                                relation_file=f"data/{task}/relation_name.txt",
                                rel_index=f"data/{task}/rel_index.npy",
                                rel_cluster_labels=f"data/{task}/CFRLdata_10_100_10_5/rel_cluster_label_0.npy")
            rel2id = sampler.rel_id
            trainsforms_data_train(config=80,
                                rel_cluster_labels=f"data/{task}/CFRLdata_10_100_10_5/rel_cluster_label_0.npy",
                                rel_index=f"data/{task}/rel_index.npy",
                                file_path=f"data/{task}/CFRLdata_10_100_10_5/train_0.txt",
                                category="train",
                                relation_name=f"data/{task}/relation_name.txt", seed=100 + rou * 100,task=task)
            trainsforms_data_text(config=80,
                                rel_cluster_labels=f"data/{task}/CFRLdata_10_100_10_5/rel_cluster_label_0.npy",
                                rel_index=f"data/{task}/rel_index.npy",
                                file_path=f"data/{task}/CFRLdata_10_100_10_5/test_0.txt",
                                category="test",
                                relation_name=f"data/{task}/relation_name.txt", seed=100 + rou * 100,task=task)
        else:
            sampler = data_sampler(config=41, seed=100 + rou * 100,
                                test_file=f"data/{task}/CFRLdata_10_100_10_5/test_0.txt",
                                relation_file=f"data/{task}/relation_name.txt",
                                rel_index=f"data/{task}/rel_index.npy",
                                rel_cluster_labels=f"data/{task}/CFRLdata_10_100_10_5/rel_cluster_label_0.npy")
            rel2id = sampler.rel_id
            trainsforms_data_train(config=41,
                                rel_cluster_labels=f"data/{task}/CFRLdata_10_100_10_5/rel_cluster_label_0.npy",
                                rel_index=f"data/{task}/rel_index.npy",
                                file_path=f"data/{task}/CFRLdata_10_100_10_5/train_0.txt",
                                category="train",
                                relation_name=f"data/{task}/relation_name.txt", seed=100 + rou * 100,task=task)
            trainsforms_data_text(config=41,
                                rel_cluster_labels=f"data/{task}/CFRLdata_10_100_10_5/rel_cluster_label_0.npy",
                                rel_index=f"data/{task}/rel_index.npy",
                                file_path=f"data/{task}/CFRLdata_10_100_10_5/test_0.txt",
                                category="test",
                                relation_name=f"data/{task}/relation_name.txt", seed=100 + rou * 100,task=task)

        for steps, (test_data, current_relations,seen_relations) in enumerate(sampler):
            if steps <=3:
                continue
            temp_rel2id = [(rel2id[x], x) for x in seen_relations]
            instruction=""
            if task=="fewrel":
                instruction = "You are a classifier.\n " \
                              "I will provide a text and two entities.\n " \
                              "Your goal is to determine the relationship between these two entities and " \
                              "select one category ID that best represents the relationship between these two entities.\n " \
                              "The categories are as follows:\n\n" + \
                              '\n'.join(f"{relation[0]}: {relation[1]}" for relation in temp_rel2id) + \
                              "It is important to note that when outputting, " \
                              "you do not need to output the questions I gave you, " \
                              "just the category ID, without any other information." + \
                              "\nNow I will provide the sentence and entities in the format of the example input. "
            if task=="tacred":
                instruction = "You are a classifier.\n " \
                              "I will provide a text and two entities.\n " \
                              "Your goal is to determine the relationship between these two entities and " \
                              "select one category ID that best represents the relationship between these two entities.\n " \
                              "The categories are as follows:\n\n" + \
                              '\n'.join(f"{relation[0]}: {relation[1]}" for relation in temp_rel2id) + \
                              "It is important to note that when outputting, " \
                              "you do not need to output the questions I gave you, " \
                              "just the category ID, without any other information."+ \
                              "\nNow I will provide the sentence and entities in the format of the example input. "
            if steps==0:
                shell_script_path = f"llama3_lora_{task}_5_1.sh"  # 替换为你的 shell 脚本路径
                result = subprocess.run(["bash", shell_script_path])
                
                if result.returncode == 0:
                    logger.info("训练脚本执行成功！")
                else:
                    logger.info(f"训练脚本执行失败，返回码：{result.returncode}")
                
                    assert 0
                save_loss_png(source_folder = f"save_dora_{task}_5" ,
                              destination_folder = f"{task}_dora_loss",task=f"{task}",number=rou,task_length=1 )
                
                text_script_path= f"llama3_lora_text_{task}_1.sh"
                result = subprocess.run(["bash", text_script_path])
                if result.returncode == 0:
                    logger.info("测试脚本执行成功！")
                else:
                    logger.info(f"测试脚本执行失败，返回码：{result.returncode}")
                    assert 0
                accuracy = eval_accuracy(f"{task}_dora_out/generated_predictions.jsonl")
                test_total.append(accuracy)
                save_shell_script_path=f"lora_adapter_{task}_1.sh"
                result = subprocess.run(["bash", save_shell_script_path])

                if result.returncode == 0:
                    logger.info("保存脚本执行成功！")
                else:
                    logger.info(f"保存脚本执行失败，返回码：{result.returncode}")
                    assert 0
            else:
                if steps == 4:
                    change_sh_text(f"llama3_lora_text_{task}_n.sh", task=task, step=j)
                    text_script_path = f"llama3_lora_text_{task}_n.sh"
                    result = subprocess.run(["bash", text_script_path])
                    if result.returncode == 0:
                        logger.info("测试脚本执行成功！")
                    else:
                        logger.info(f"测试脚本执行失败，返回码：{result.returncode}")
                        assert 0
                    accuracy = eval_accuracy(f"{task}_dora_out/generated_predictions.jsonl")
                    test_total.append(accuracy)
                    save_shell_script_path = f"lora_adapter_{task}_n.sh"
                    result = subprocess.run(["bash", save_shell_script_path])

                    if result.returncode == 0:
                        logger.info("保存脚本执行成功！")
                    else:
                        logger.info(f"保存脚本执行失败，返回码：{result.returncode}")
                        assert 0
                else: 
                    change_sh(f"llama3_lora_{task}_5_n.sh",task=task,step=j)
                    shell_script_path = f"llama3_lora_{task}_5_n.sh"  # 替换为你的 shell 脚本路径
                    result = subprocess.run(["bash", shell_script_path])
                    
                    if result.returncode == 0:
                        logger.info("训练脚本执行成功！")
                    else:
                        logger.info(f"训练脚本执行失败，返回码：{result.returncode}")
                        assert 0
                    save_loss_png(source_folder=f"save_dora_{task}_5",
                                destination_folder=f"{task}_dora_loss", task=f"{task}",
                                number=rou, task_length=j)
                    
                    change_sh_text(f"llama3_lora_text_{task}_n.sh", task=task, step=j)
                    
                    text_script_path = f"llama3_lora_text_{task}_n.sh"
                    result = subprocess.run(["bash", text_script_path])
                    if result.returncode == 0:
                        logger.info("测试脚本执行成功！")
                    else:
                        logger.info(f"测试脚本执行失败，返回码：{result.returncode}")
                        assert 0
                    accuracy = eval_accuracy(f"{task}_dora_out/generated_predictions.jsonl")
                    test_total.append(accuracy)
                    save_shell_script_path = f"lora_adapter_{task}_n.sh"
                    result = subprocess.run(["bash", save_shell_script_path])

                    if result.returncode == 0:
                        logger.info("保存脚本执行成功！")
                    else:
                        logger.info(f"保存脚本执行失败，返回码：{result.returncode}")
                        assert 0
                j+=1
            logger.info(f"{task} rou:{rou} step:{steps} total_acc:{test_total}")
        result_whole_test.append(np.array(test_total) * 100)
        avg_result_all_test = np.average(result_whole_test, 0)
        std_result_all_test = np.std(result_whole_test, 0)
        logger.info("result_whole_test")
        logger.info(result_whole_test)
        logger.info("avg_result_all_test")
        logger.info(avg_result_all_test)
        logger.info("std_result_all_test")
        logger.info(std_result_all_test)



        # for i in range(2,9):
        #     with open(file="/home/xurf23/xurf_project/LLaMA-Factory-main/LLaMA-Factory-main/llama3_lora_fewrel_5_n.sh",mode="r+")as file:





# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # logger.add("log_{time:YYYY-MM-DD-HH-mm-ss}_fewrel_5_rslora.log", format="{time} - {level} - {message}",
    #            level="INFO")
    logger.add("log_2025-06-23-10-28-02_fewrel_5_rslora.log", format="{time} - {level} - {message}", level="INFO", mode="a")
    main("fewrel")
    

    # fewrel="fewrel"
    # with open(file="llama3_lora_text_fewrel_1.sh", mode="r") as file:
    #     line = file.readlines()
    #     print(line[7])
    #     line[7] = f"    --eval_dataset fewrel_text_5_1 \\\n"
    # with open(file="llama3_lora_text_fewrel_1.sh", mode="w") as file:
    #     file.writelines(line)

