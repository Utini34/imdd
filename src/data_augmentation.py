import pandas as pd

from datasets import ClassLabel, Dataset, Features, Value, concatenate_datasets

from eda import eda


class DataAugmentation:
    def __init__(self, data):
        self.data = data

    def eda(self, **eda_kwargs):
        new_examples = []
        for example in self.data:
            # Apply the data_aug function to the input sentence
            augmented_sentences = eda(example["text"], **eda_kwargs)
            for sentence in augmented_sentences:
                new_example = example.copy()
                new_example["text"] = sentence
                new_examples.append(new_example)

        # data schema
        class_label = ClassLabel(names=['neg', 'pos'])
        schema = Features(
            {
                'text': Value('string'),  # Example feature
                'label': class_label,  # Assign the ClassLabel to the 'label' feature
            }
        )
        augmented_dataset = Dataset.from_pandas(
            pd.DataFrame(new_examples), features=schema
        )
        # Combine the new examples into an extended dataset
        extended_dataset = concatenate_datasets([self.data, augmented_dataset])

        return extended_dataset

    def gan(self, num_generated_samples):
        pass
