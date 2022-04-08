import os

from a_configuration.a_base_config.config_parse import S3_ACCESS_ID, S3_ACCESS_SECRET

import boto3


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


def upload_file(name, local, obj):
    bucket_name = name
    s3 = boto3.resource('s3', aws_access_key_id=S3_ACCESS_ID, aws_secret_access_key=S3_ACCESS_SECRET)
    bucket = s3.Bucket(bucket_name)

    local_file = local
    obj_file = obj
    bucket.upload_file(local_file, obj_file)


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


def load_instance(bucket_name, flie_path):
    client = boto3.resource('s3', aws_access_key_id=S3_ACCESS_ID, aws_secret_access_key=S3_ACCESS_SECRET)
    obj = client.Object(bucket_name, flie_path)

    myBody = obj.get()['Body'].read()
    myBody = myBody.decode()

    print(myBody)


if __name__ == '__main__':
    #show_bucket_name()
    bucket_name = 'linklab'
    flie_path = 'knapsack_instances/FI/instances/n_50_wp_12.5/instance0.csv'
    load_instance(bucket_name, flie_path)