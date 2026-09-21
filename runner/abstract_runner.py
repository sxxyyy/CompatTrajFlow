import logging
from argparse import Namespace

from numpy import ndarray
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Generator
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class AbstractRunner:
    """Abstract Runner class."""

    def __init__(self, args: Namespace, generator: Generator) -> None:
        """
        Initialize the AbstractRunner.

        Args:
            args (Namespace): Arguments for the runner.
            generator (Generator): Torch random generator.
        """
        self.location = args.location
        self.device = args.device
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_edges = args.num_edges + 2  # Adjust for padding and end tokens
        self.num_times = args.num_times
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.force_train = args.force_train
        self.use_amp = args.use_amp
        self.batch_log_interval = args.batch_log_interval
        self.test_log_interval = args.test_log_interval
        self.checkpoint_path = "checkpoints"
        self.generator = generator
        logger.info("Use Automatic Mixed Precision: %s", args.use_amp)

        self.patience = args.patience
        self.relative_delta = args.relative_delta
        self.best_loss = float("inf")
        self.counter = 0

        self.global_step = 0
        self.total_train_loss = 0.0

        self.anomaly_type = None

    def __call__(
        self,
        state: str,
        dataloader: DataLoader | None,
        val_dataloader: DataLoader | None,
        anomaly_dataloader_dict: dict[str, DataLoader],
    ) -> list[dict[str, float]] | None:
        """
        Run the runner in either training or testing mode.

        Args:
            state (str): "train" to train the model, "test" to evaluate.
            dataloader (DataLoader): DataLoader for training or testing.
            val_dataloader (DataLoader, optional): DataLoader for validation during training.

        Returns:
            list[float] | None: List of average precision scores for validation or test, or None if training is skipped.

        Raises:
            AssertionError: If state is not "train" or "test", or if validation dataloader is missing during training.
        """

        assert state in ["train", "test"], "The state must set within train or test"
        match state:
            case "train":
                assert dataloader is not None, (
                    "Training Dataloader must be provided for training"
                )
                return self.train_step(
                    dataloader, val_dataloader, anomaly_dataloader_dict
                )
            case "test":
                return [self.test_step(anomaly_dataloader_dict)]
            case _:
                return None

    def train_step(
        self,
        dataloader: DataLoader,
        val_dataloader: DataLoader,
        anomaly_dataloader_dict: dict[str, DataLoader],
    ) -> list[dict[str, float]] | None:
        if self.is_checkpoint_exists() and not self.force_train:
            logger.info("Checkpoint exists. Resume training from checkpoint.")
            self.load_checkpoint()
        if self.force_train:
            logger.info("Checkpoint exists. Overwrite it with new training.")
        self.create_checkpoint_dir()
        self.global_step = 0
        self.total_train_loss = 0.0
        self.best_loss = float("inf")
        self.counter = 0
        validation_result = []
        for i in range(self.epoch):
            logger.info("Start epoch %d", i + 1)
            if hasattr(self, "val") and callable(getattr(self, "val")):
                self.train(dataloader)
                val_loss = self.val(val_dataloader)
                early_stop = self.early_stopping(val_loss)
            else:
                epoch_avg_loss = self.train(dataloader)
                logger.info(
                    "Epoch %d finished: epoch_avg=%.4f, global_avg=%.4f, global_step=%d",
                    i + 1,
                    epoch_avg_loss,
                    self.total_train_loss / self.global_step
                    if self.global_step > 0
                    else float("inf"),
                    self.global_step,
                )
                early_stop = self.early_stopping_epoch(i + 1, epoch_avg_loss)

            if early_stop:
                logger.info(
                    "Early stopping triggered at epoch %d. Stopping training.", i + 1
                )
                break
        self.save_checkpoint()
        test_result = self.get_test_result(anomaly_dataloader_dict)
        validation_result.append(test_result)
        self.free_vram()
        return validation_result

    def test_step(
        self,
        anomaly_dataloader_dict: dict[str, DataLoader],
    ) -> dict[str, float]:
        self.load_checkpoint()
        logger.info("Start testing")

        test_result = self.get_test_result(anomaly_dataloader_dict)
        self.free_vram()
        return test_result

    def get_test_result(
        self, anomaly_dataloader_dict: dict[str, DataLoader]
    ) -> dict[str, float]:
        anomaly_result = {}
        for (
            anomaly_type,
            anomaly_dataloader,
        ) in anomaly_dataloader_dict.items():
            logger.info("Start testing anomaly type: %s", anomaly_type)
            self.anomaly_type = anomaly_type[:-4]  # for fotraj only
            y_true, y_pred = self.test(anomaly_dataloader)
            metrice_score_dict = self.get_scores_with_different_metrices(y_true, y_pred)
            anomaly_result[anomaly_type] = metrice_score_dict
            for key, value in metrice_score_dict.items():
                logger.info(
                    "Anomaly Type: %s, Metric: %s, Score: %.4f",
                    anomaly_type,
                    key,
                    value,
                )
        return anomaly_result

    def get_scores_with_different_metrices(
        self, y_true: ndarray, y_pred: ndarray
    ) -> dict[str, float]:
        """
        Calculate various scores based on the true and predicted values.

        Args:
            y_true (ndarray): Ground truth (correct) target values.
            y_pred (ndarray): Estimated probabilities or decision function.

        Returns:
            dict[str, float]: Dictionary containing average precision and AUC scores.
        """
        return {
            "Average Precision": self.get_average_precision_score(y_true, y_pred),
            "ROC_AUC": self.get_auc_score(y_true, y_pred),
        }

    def train(self, dataloader: DataLoader) -> float:
        """
        Abstract train method for controlling all model training procedures.

        Args:
            dataloader (DataLoader): DataLoader for training data.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def test(self, dataloader: DataLoader) -> tuple[ndarray, ndarray]:
        """
        Run the test phase using the provided dataloader.

        Args:
            dataloader (DataLoader): DataLoader containing the test dataset.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple of ground truth labels and predicted values.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """

        raise NotImplementedError

    def create_checkpoint_dir(self):
        """
        Create the checkpoint path for saving the model.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """

        raise NotImplementedError

    def save_checkpoint(self):
        """
        Save the module state dict to the checkpoint path.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """

        raise NotImplementedError

    def load_checkpoint(self):
        """
        Load the module state dict from the checkpoint path.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """

        raise NotImplementedError

    def is_checkpoint_exists(self):
        """
        Check if the checkpoint exists.

        Returns:
            bool: True if checkpoint exists, False otherwise.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """

        raise NotImplementedError

    def free_vram(self):
        """Free vram for next training or testing"""
        raise NotImplementedError

    def get_average_precision_score(self, y_true: ndarray, y_pred: ndarray) -> float:
        """
        Calculate the average precision score.

        Args:
            y_true (ndarray): Ground truth (correct) target values. Must be a 1D array.
            y_pred (ndarray): Estimated probabilities or decision function. Must be a 1D array.

        Returns:
            float: Average precision score, a value between 0 and 1.

        Raises:
            AssertionError: If the shapes of y_true and y_pred do not match or if they are not 1D arrays.
        """
        assert y_true.shape == y_pred.shape, (
            "Shape of y_true and y_pred must be the same"
        )
        assert y_true.ndim == 1 and y_pred.ndim == 1, (
            "y_true and y_pred must be 1D arrays"
        )
        return float(average_precision_score(y_true, y_pred))

    def get_auc_score(self, y_true: ndarray, y_pred: ndarray) -> float:
        """
        Calculate the AUC score.

        Args:
            y_true (ndarray): Ground truth (correct) target values. Must be a 1D array.
            y_pred (ndarray): Estimated probabilities or decision function. Must be a 1D array.

        Returns:
            float: AUC score, a value between 0 and 1.

        Raises:
            AssertionError: If the shapes of y_true and y_pred do not match or if they are not 1D arrays.
        """
        assert y_true.shape == y_pred.shape, (
            "Shape of y_true and y_pred must be the same"
        )
        assert y_true.ndim == 1 and y_pred.ndim == 1, (
            "y_true and y_pred must be 1D arrays"
        )
        return float(roc_auc_score(y_true, y_pred))

    def log_batch_loss(self, batch_index: int, num_batch: int, loss: float):
        """
        Log the loss value at specified batch intervals during training.

        Args:
            batch_index (int): The index of the current batch within the epoch.
            num_batch (int): The total number of batches in the epoch.
            loss (float): The loss value for the current batch.
        """
        read_index = batch_index + 1
        read_num_batch = num_batch
        if read_index % self.batch_log_interval == 0 or read_index == read_num_batch:
            logger.info("Batch %d/%d, Loss: %.4f", read_index, read_num_batch, loss)

    def early_stopping_epoch(self, epoch: int, loss: float) -> bool:
        """
        Epoch-level early stopping based on training (or validation) loss.
        Compares the current epoch's loss against the best loss seen so far.

        Args:
            epoch (int): The current epoch number (1-indexed).
            loss (float): The epoch-level loss (epoch average or validation loss).

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.best_loss == float("inf"):
            self.best_loss = loss
            logger.info("Initial best loss: %.4f at epoch %d.", self.best_loss, epoch)
            return False

        relative_improvement = (self.best_loss - loss) / abs(self.best_loss)
        if loss < self.best_loss and relative_improvement > self.relative_delta:
            self.best_loss = loss
            self.counter = 0
            logger.info(
                "Epoch %d: loss improved to %.4f (relative: %.2f%%).",
                epoch,
                self.best_loss,
                relative_improvement * 100,
            )
        else:
            self.counter += 1
            logger.info(
                "Epoch %d: no improvement (best=%.4f, current=%.4f, rel=%.2f%%), patience=%d/%d.",
                epoch,
                self.best_loss,
                loss,
                relative_improvement * 100,
                self.counter,
                self.patience,
            )
            if self.counter >= self.patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                return True
        return False

    def log_test_progress(self, batch_index: int, num_batch: int):
        """
        Log the progress of the test phase at specified intervals.

        Args:
            batch_index (int): The index of the current batch within the test set.
            num_batch (int): The total number of batches in the test set.
        """
        read_index = batch_index + 1
        read_num_batch = num_batch
        if read_index % self.test_log_interval == 0 or read_index == read_num_batch:
            logger.info("Test Batch %d/%d", read_index, read_num_batch)

    def get_score(self, test_dataloader: DataLoader):
        true, pred = self.test(test_dataloader)
        score = float(roc_auc_score(true, pred))
        return score

    def __str__(self) -> str:
        """
        Return a string representation of the runner.

        Returns:
            str: String with embedding and hidden size.
        """
        return f"{self.embedding_size}_{self.hidden_size}"
