import pickle as pkl
from argparse import ArgumentParser
from logging import getLogger

from tqdm import tqdm

logger = getLogger(__name__)


class DatasetClassConvertor:
    def __init__(self, pkls_path: list):
        self.pkls_path = pkls_path
        self.data_list = []
        self.dataset = self.load_pkl(self.pkls_path[0])
        self.load_and_merge_pkls()
        self.bbox_label_3d = "bbox_label_3d"
        self.classes_ids = self.dataset["metainfo"]["categories"]
        print(f"classes_ids: {self.classes_ids}")
        self.ids_classes = {v: k for k, v in self.classes_ids.items()}
        print(f"ids_classes: {self.ids_classes}")

    @staticmethod
    def load_pkl(pkl_path):
        with open(pkl_path, "rb") as f:
            dataset = pkl.load(f)
        return dataset

    def load_and_merge_pkls(self):
        for pkl_path in tqdm(self.pkls_path, desc="Loading and merging pkls"):
            dataset = self.load_pkl(pkl_path)
            data_list = dataset["data_list"]
            self.data_list.extend(data_list)
        self.dataset["data_list"] = self.data_list

    def best_skeleton(self, num_batches, limit=None, sort_mode="categories", target_labels=None):
        """
        Parameters:
        -----------
        num_batches : int
            Number of batches to split the data into
        limit : int, optional
            Maximum number of samples to keep
        sort_mode : str
            Sorting mode: "categories" for most unique categories,
                        "specific_labels" for most instances of specific labels
        target_labels : list or set, optional
            Target labels to prioritize when sort_mode is "specific_labels"
        """
        print(30*"=")
        print(f"Number of batches = {num_batches}\nNumber of key frames = {limit}")
        print(f"Sort mode: {sort_mode}")
        if sort_mode == "specific_labels" and target_labels:
            target_labels = set(target_labels)  # Convert to set for faster lookup
            target_names = [self.ids_classes[label] for label in target_labels]
            print(f"Target labels: {target_labels}")
            print(f"Target names: {target_names}")
        
        # Filter samples with no instances
        self.data_list = filter(lambda x: len(x["instances"]) > 0, self.data_list)
        self.data_list = list(self.data_list)
        print(f"Initial filtered data_list length: {len(self.data_list)}")

        # Define sorting key based on mode
        if sort_mode == "specific_labels" and target_labels:
            sort_key = lambda x: (
                self.count_instances_with_labels(x, target_labels),  # Primary sort by target labels count
                len(x["instances"])  # Secondary sort by total instances
            )
        elif sort_mode == "instances_and_labels" and target_labels:
            sort_key = lambda x: (
                len(x["instances"]),  # Primary sort by total instances
                self.count_instances_with_labels(x, target_labels)  # Secondary sort by target labels count
            )
        elif sort_mode == "categories_and_labels" and target_labels:
            sort_key = lambda x: (
                self.get_unique_categories(x),  # Primary sort by unique categories
                self.count_instances_with_labels(x, target_labels)  # Secondary sort by target labels count
            )
        else:  # default to categories mode
            sort_key = lambda x: (
                self.get_unique_categories(x),  # Primary sort by unique categories
                len(x["instances"])  # Secondary sort by total instances
            )

        # Sort the entire dataset
        self.data_list.sort(key=sort_key, reverse=True)

        # Split into batches
        batch_size = max(1, (len(self.data_list) + num_batches - 1) // num_batches)
        batches = []
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.data_list))
            if start_idx < len(self.data_list):
                batches.append(self.data_list[start_idx:end_idx])

        # Apply limit to each batch if specified
        each_batch_limit = limit // num_batches if limit is not None else None
        limited_batches = []
        for batch in batches:
            if each_batch_limit is not None:
                batch = batch[:each_batch_limit]
            limited_batches.append(batch)

        # Combine all batches back into data_list
        self.data_list = []
        for batch in limited_batches:
            self.data_list.extend(batch)

        print(
            f"Final data_list length after splitting, sorting, and limiting: {len(self.data_list)}"
        )

        # Store in dataset
        self.dataset["data_list"] = self.data_list

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pkl.dump(self.dataset, f)
        
    def get_instance_class(self, instance):
        return instance[self.bbox_label_3d]     

    def counter(self):
        classes_counter = {}
        for sample in tqdm(self.data_list):
            for instance in sample["instances"]:
                class_id = self.get_instance_class(instance)
                if class_id not in classes_counter:
                    classes_counter[class_id] = 0
                classes_counter[class_id] += 1
        print(30*"-")
        for class_id, count in classes_counter.items():
            print(f"{self.ids_classes[class_id]}: {count}")
        print(30*"=")


    def get_unique_categories(self, sample):
        """Helper method to count unique categories in a sample"""
        categories = set()
        for instance in sample["instances"]:
            categories.add(self.get_instance_class(instance))
        return len(categories)
    def count_instances_with_label(self, sample, target_label):
        """Helper method to count instances of a specific label in a sample"""
        return sum(1 for instance in sample["instances"] 
                if self.get_instance_class(instance) == target_label)
    def count_instances_with_labels(self, sample, target_labels):
        """Helper method to count instances of specific labels in a sample
        
        Parameters:
        -----------
        sample : dict
            Sample data containing instances
        target_labels : list or set
            List of target labels to count
            
        Returns:
        --------
        int : Total count of instances with any of the target labels
        """
        return sum(1 for instance in sample["instances"] 
                if self.get_instance_class(instance) in target_labels)
def convert_dataset(pkls_dir, save_path, num_batches, limit, rule, target_labels):
    convertor = DatasetClassConvertor(pkls_dir)
    convertor.best_skeleton(limit=limit, num_batches=num_batches,sort_mode= rule, target_labels=target_labels)
    convertor.counter()
    convertor.save(save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    import ast 
    
    from pkls_path import pkls_path

    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the merged pkl"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the number of files to process"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches to split the data into",
    )
    
    parser.add_argument(
        "--rule",
        type=str,
        default='categories',
        help="Rule of selecting skeeleton categoires",
    )
    parser.add_argument(
        "--target-labels",
        default=[4],
        help="Target labels if specefic labels rule selected",
    )
    args = parser.parse_args()
    if not isinstance(args.target_labels, list):
        args.target_labels = ast.literal_eval(args.target_labels)
    convert_dataset(
        pkls_path, args.save_path, limit=args.limit, num_batches=args.num_batches,
        rule = args.rule, target_labels = args.target_labels
    )
