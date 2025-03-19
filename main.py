if __name__ == "__main__":

    __import__("warnings").filterwarnings("ignore")
    __import__("os").environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    from config import Config
    from data.data_loader import ShakespeareDataLoader
    from models.model import ShakespeareTextGeneratorModel
    from models.trainer import ShakespeareTextGeneratorTrainer
    from models.generate import ShakespeareTextGenerator

    config = Config()

    data_loader = ShakespeareDataLoader(config=config)
    dataset = data_loader.get_dataset()

    model = ShakespeareTextGeneratorModel(config=config, vocab_size=len(data_loader.vocab))

    trainer = ShakespeareTextGeneratorTrainer(model=model, dataset=dataset, config=config)
    history = trainer.train()

    generator = ShakespeareTextGenerator(model, data_loader.char2idx, data_loader.idx2char)
    print(generator.generate("ROMEO: ", temperature=0.7))
