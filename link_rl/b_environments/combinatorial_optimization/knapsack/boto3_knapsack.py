import os

from link_rl.a_configuration.a_base_config.config_parse import S3_ACCESS_ID, S3_ACCESS_SECRET

import boto3 #conda install boto3


def show_bucket_name():
    s3 = boto3.resource('s3', aws_access_key_id=S3_ACCESS_ID, aws_secret_access_key=S3_ACCESS_SECRET)

    for bucket in s3.buckets.all():
        print(bucket.name)


def download_file(name, obj, save):
    bucket_name = name
    s3 = boto3.resource('s3', aws_access_key_id=S3_ACCESS_ID, aws_secret_access_key=S3_ACCESS_SECRET)
    bucket = s3.Bucket(bucket_name)

    obj_file = obj
    save_file = save

    bucket.download_file(obj_file, save_file)


def upload_file(name, content, obj):
    bucket_name = name
    s3 = boto3.client('s3', aws_access_key_id=S3_ACCESS_ID, aws_secret_access_key=S3_ACCESS_SECRET)
    encode_file = str(content)
    s3.put_object(Bucket=bucket_name, Key=obj, Body=encode_file)


def hard_instacnces_upload():
    bucket_name = 'linklab'

    instance_info_keys = ["n_50_r_100", "n_300_r_600", "n_500_r_1000"] #hard_instances_key

    INDEX = 3
    M = 1000

    for idx in range(INDEX):
        for m in range(M):
            file_path = os.path.join("hard_instances", instance_info_keys[idx])

            if not os.path.isdir(file_path):
                os.makedirs(file_path)

            file_name = '/instance' + str(m) + ".csv"
            local_file_path = file_path + "/instance" + str(m) + ".csv"
            obj_file_name = 'knapsack_instances/HI/instances/' + str(instance_info_keys[idx]) + file_name
            upload_file(bucket_name, local_file_path, obj_file_name)
            print(file_name)


def random_instacnces_upload():
    bucket_name = 'linklab'

    instance_info_keys = ["n_50_r_100", "n_300_r_600", "n_500_r_1800"] #random_instances_key

    INDEX = 3
    M = 1000

    for idx in range(INDEX):
        for m in range(M):
            file_path = os.path.join("random_instances", instance_info_keys[idx])

            if not os.path.isdir(file_path):
                os.makedirs(file_path)

            file_name = '/instance' + str(m) + ".csv"
            local_file_path = file_path + "/instance" + str(m) + ".csv"
            obj_file_name = 'knapsack_instances/RI/instances/' + str(instance_info_keys[idx]) + file_name
            upload_file(bucket_name, local_file_path, obj_file_name)
            print(file_name)


def fixed_instacnces_upload():
    bucket_name = 'linklab'

    instance_info_keys = ["n_50_wp_12.5", "n_300_wp_37.5", "n_500_wp_37.5"] #fixed_instances_key

    INDEX = 3
    M = 1000

    for idx in range(INDEX):
        for m in range(M):
            file_path = os.path.join("fixed_instances", instance_info_keys[idx])

            if not os.path.isdir(file_path):
                os.makedirs(file_path)

            file_name = '/instance' + str(m) + ".csv"
            local_file_path = file_path + "/instance" + str(m) + ".csv"
            obj_file_name = 'knapsack_instances/FI/instances/' + str(instance_info_keys[idx]) + file_name
            upload_file(bucket_name, local_file_path, obj_file_name)
            print(file_name)


def load_instance(bucket_name, file_path):
    client = boto3.resource('s3', aws_access_key_id=S3_ACCESS_ID, aws_secret_access_key=S3_ACCESS_SECRET)
    obj = client.Object(bucket_name, file_path)

    myBody = obj.get()['Body'].read()
    myBody = myBody.decode()

    info = ""
    info_list = []
    for x in myBody:
        if x == ",":
            info_list.append(info)
            info = ""
        elif x == '\n' or x == '\r':
            info_list.append(info)
            info = ""
        else:
            info += x

    data = []

    for y in info_list:
        if y != "":
            data.append(y)

    num_items = int(len(data)/2) - 1
<<<<<<< HEAD:link_rl/b_environments/combinatorial_optimization/knapsack/boto3_knapsack.py

    # state = np.zeros(shape=(num_items+4, 2), dtype=float)
    #
    # for item_idx in range(num_items):
    #     state[item_idx + 4][0] = data[item_idx * 2]
    #     state[item_idx + 4][1] = data[item_idx * 2 + 1]

    # state[0][1] = data[-1]

    items = []
    for item_idx in range(num_items):
        items.append([float(data[item_idx * 2]), float(data[item_idx * 2 + 1])])
=======
    state = np.zeros(shape=(num_items+4, 2), dtype=float)

    data_idx = 4

    for item_idx in range(num_items):
        state[data_idx][0] = data[item_idx + data_idx - 4]
        state[data_idx][1] = data[item_idx + data_idx + 1 - 4]

        data_idx += 1

    state[0][1] = data[-1]
>>>>>>> f3684a1ee26789bc76c24ee72e30e96bb0d523ba:b_environments/combinatorial_optimization/boto3_knapsack.py

    #print(data)
    #print(state)

    return items, float(data[-1])


def load_solution(bucket_name, file_path):
    client = boto3.resource('s3', aws_access_key_id=S3_ACCESS_ID, aws_secret_access_key=S3_ACCESS_SECRET)
    obj = client.Object(bucket_name, file_path)

    myBody = obj.get()['Body'].read()
    myBody = myBody.decode()

    info = ""
    info_list = []
    for x in myBody:
        if x == '\n' or x == '\r':
            info_list.append(info)
            break
        elif x == " ":
            pass
        else:
            info += x

    data = []

    for y in info_list:
        if y != "":
            data.append(y)

    return float(data[0])


def check_link_solution(bucket_name, file_path):
    client = boto3.resource('s3', aws_access_key_id=S3_ACCESS_ID, aws_secret_access_key=S3_ACCESS_SECRET)
    obj = client.Object(bucket_name, file_path)

    myBody = obj.get()['Body'].read()
    myBody = myBody.decode()

    info = ""
    info_list = []
    for x in myBody:
        if x == '[' or x == " ":
            pass
        elif x == ',' or x == ']':
            info_list.append(info)
            info = ""
        else:
            info += x

    print(info_list)

    return float(info_list[-1])

def check_solution(bucket_name, file_path, size):
    total = 0
    for i in range(size):
        file_paths = file_path + '/link_solution' + str(i) + '.csv'
        value = check_link_solution(bucket_name, file_paths)

        if value == 1.0:
            total += 1

    print("result : {0}%".format((total/size)*100))


if __name__ == '__main__':
    #show_bucket_name()
    bucket_name = 'linklab'
    file_path = 'knapsack_instances/RI/instances/n_50_r_100/instance0.csv'
    #file_path = 'knapsack_instances/FI/instances/n_50_wp_12.5/instance0.csv'
    load_instance(bucket_name, file_path)

    #file_path = 'knapsack_instances/RI/optimal_solution/n_50_r_100/solution0.csv'
    #load_solution(bucket_name, file_path)

    # file_path = 'knapsack_instances/TEST/link_solution/2022422zerowin'
    # check_solution(bucket_name, file_path, 3)
