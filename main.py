if __name__ == "__main__":

    __import__('warnings').filterwarnings('ignore')
    __import__('os').environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

    from config import Config
    from data.data_loader import ShakespeareDataLoader

    config = Config()
    
    data_loader = ShakespeareDataLoader(config=config)
    dataset = data_loader.get_dataset()

    
    