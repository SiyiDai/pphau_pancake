from tensorflow.keras import models
from tensorflow.keras import layers
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from model.data_loader import get_data
from model.params import (
    sample_weight,
    classes,
    params_model,
    params_fit,
)
import pickle

model_save_name = "finalized_model.sav"


class LSTMModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        """initiate Class object"""
        self.model = None
        self.history = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train

    def initialize_model(self):
        """instanciate, compile and return the LSTM model"""
        model = models.Sequential(name="LSTM_Model")

        model.add(
            layers.LSTM(
                params_model["0_LSTM"],
                input_shape=params_model["0_input_shape"],
                return_sequences=True,
                name="0_LSTM",
            )
        )
        model.add(
            layers.Dropout(
                rate=params_model["1_dropout"],
                name="1_Dropout",
            )
        )

        # model.add(
        #     layers.LSTM(
        #         params_model["2_LSTM"],
        #         return_sequences=True,
        #         name="2_LSTM",
        #     )
        # )
        # # model.add(layers.RepeatVector(params_model["3_repeat_vector"]))
        # model.add(
        #     layers.LSTM(
        #         params_model["3_LSTM"],
        #         activation="tanh",
        #         return_sequences=True,
        #         name="3_LSTM",
        #     )
        # )

        # model.add(
        #     layers.LSTM(
        #         params_model["4_LSTM"],
        #         return_sequences=True,
        #         name="4_LSTM",
        #     )
        # )

        # model.add(
        #     layers.Dropout(
        #         rate=params_model["5_dropout"],
        #         name="5_Dropout",
        #     ),
        # )
        # model.add(
        #     layers.TimeDistributed(
        #         layers.Dense(
        #             params_model["6_time_distributed"],
        #         ),
        #         name="6_TimeDistributed",
        #     )
        # )

        model.add(
            layers.Flatten(
                name="2_Flatten",
            ),
        )
        model.add(
            layers.Dense(
                params_model["3_dense"],
                activation="relu",
                name="3_Dense",
            )
        )
        model.add(
            layers.Dense(
                params_model["4_dense"],
                activation="softmax",
                name="4_Dense",
            ),
        )

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self.model = model

    def model_fit(self):
        """fit the model"""

        es = EarlyStopping(patience=params_fit["patience"], restore_best_weights=True)

        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=params_fit["validation_split"],
            epochs=params_fit["epochs"],
            # sample_weight=sample_weight,
            batch_size=params_fit["batch_size"],
            callbacks=[es],
            verbose=params_fit["verbose"],
        )
        self.history = history
        return history

    def evaluate(self):
        """evaluates the model on test data and return accuracy"""
        evaluation = self.model.evaluate(self.X_test, self.y_test)
        accuracy = evaluation[1]
        return accuracy

    def save(self):
        # save the model to disk
        filename = model_save_name
        pickle.dump(self.model, open(filename, "wb"))

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def load(self):
        # load the model
        filename = model_save_name
        model = pickle.load(open(filename, "rb"))
        return model

    def show_loss(self):
        loss_history = self.history.history["loss"]

        plt.plot(loss_history)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()

    def show_val_loss(self):
        loss_history = self.history.history["val_loss"]

        plt.plot(loss_history)
        plt.xlabel("epochs")
        plt.ylabel("val_loss")
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        labels = list(classes.values())
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
        labels_name = list(classes.keys())

        df_cm = pd.DataFrame(
            cm,
            index=[i for i in list(labels_name)],
            columns=[i for i in list(labels_name)],
        )

        plt.figure(figsize=(15, 15))
        sns.heatmap(df_cm, annot=True, cmap="BuPu")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    lstm = LSTMModel(X_train, X_test, y_train, y_test)
    lstm.initialize_model()
    history = lstm.model_fit()
    res = lstm.evaluate()
    print(f"accuracy: {res}")
