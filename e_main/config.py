from a_configuration.b_single_config.unity.config_walker import ConfigWalkerSac
from g_utils.commons import print_basic_info

config = ConfigWalkerSac()
config.USE_WANDB = True
config.PLAY_MODEL_FILE_NAME = "22.3_2.3_2022_2_4_UnityWalker_SAC.pth"
config.NO_TEST_GRAPHICS = False

if __name__ == "__main__":
    print_basic_info(config=config)
