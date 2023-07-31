"""
######## CLASSIFICATION TASK ########

:Author: Pablo Sánchez Cabrera
:email: psancabrera@gmail.com

"""

import pandas as pd
import logging
from beartype import beartype
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from src.majorel._utils import IntOrNone, ListOrNone

logger = logging.getLogger(__name__)


class ClassificationTask:
    @beartype
    def __init__(self,
                 task: str = "sentiment-analysis",
                 model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Class to determine two different classification task in NLP \n
        - Determine the topic of the text \n
        - Predict the sentiment of the text \n

        In both cases, pretrained models will be used; although,
        these have been designed to respond to two different objectives

        - TOPIC \n

        A pretrained BERT model is used for this task.
        This model is a multilanguage model for using pre-trained NLI models
        as a ready-made zero-shot sequence classifiers.

        See more: Yin et al (2019). BART: Denoising Sequence-to-Sequence Pre-training
        for Natural Language Generation, Translation, and Comprehension \n
        https://towardsdatascience.com/understanding-zero-shot-learning-making-ml-more-human-4653ac35ccab

        - SENTIMENT ANALYSIS \n

        A pretrained BERT model is used for this task.
        This model is a multilanguage model (English, Dutch, German, French, Italian and Spanish)
        based on (Devlin, 2018) and was developed for sentiment analysis on product reviews.
        Try to predict the sentiment of a review in 5 categories (from 1 to 5 stars).

        See more: Devlin, C. L. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language. CoRR


        Parameters
        ----------
        task : str
            Indicate the task to be carried out. `zero-shot-classification` by default
        model_name : str
            Indicate the pretained model used. `nlptown/bert-base-multilingual-uncased-sentiment` by default

        Raises
        ------
        ValueError
            Type of task not available. Use `sentiment-analysis` or `zero-shot-classification`
        ValueError
            Type of model currently available.
            Use `nlptown/bert-base-multilingual-uncased-sentiment` or `facebook/bart-large-mnli`
        ValueError
            Incorrect model for sentimental analysis. Use `nlptown/bert-base-multilingual-uncased-sentiment`
        ValueError
            Incorrect model for topic prediction. Use `facebook/bart-large-mnli`
        ValueError
            `delete_row_character` must be greater than 0
        """
        self.task = task
        self.model_name = model_name

        self.available_task = ["sentiment-analysis", "zero-shot-classification"]
        if self.task not in self.available_task:
            raise ValueError("Type of task not available. "
                             f"Please, introduce {self.available_task}")

        self.check_model = ["nlptown/bert-base-multilingual-uncased-sentiment",
                            "facebook/bart-large-mnli"]

        if self.model_name not in self.check_model:
            raise ValueError("Model not model currently available. "
                             "Please, check the correct multilingual model "
                             "according to the classification task to be carried out. "
                             f"Available: {self.check_model}")

        if self.task == self.available_task[0] and self.model_name != self.check_model[0]:
            raise ValueError(f"Incorrect model for `{self.available_task[0]}`. "
                             f"Only available: `{self.check_model[0]}`")

        if self.task == self.available_task[1] and self.model_name != self.check_model[1]:
            raise ValueError(f"Incorrect model for `{self.available_task[1]}`. "
                             f"Only available: `{self.check_model[1]}`")

        logging.info("Define parameters for classification task. Catch errors associated")

    @beartype
    def get_task(self,
                 X: pd.DataFrame,
                 name: str,
                 split_num: IntOrNone = None,
                 labels_topic: ListOrNone = None,
                 delete_row_character: IntOrNone = 15):
        """
        Get topic of a tweet.

        Due to the processing time, the user is allowed to process to predict
        the sentiment of a collection of tweets
        chosen by the user (top x tweets - for testing).

        It also has the ability to delete records whose
        text does not meet a minimum number of characters

        Parameters
        ----------
        X : pandas dataframe
            Original dataset
        name : str
            Column name
        split_num : IntOrNone
            Used to analyze the sentiment of a specific number
            of tweets (lower than the sample). Default is None
        labels_topic : ListOrNone
            List with topic to analyse
        delete_row_character : IntOrNone
            Delete row with number of characters less than a specific value by user.
            If value is None, no row will be deleted

        Returns
        -------
        X : pandas dataframe
            Output with the topic and probabilities

        Raises
        ------
        ValueError
            Variable not available in the original dataframe
        TypeError
            Selected variable is not of type string
        ValueError
            If split_num is not None, its value must be greater than 0
         TypeError
            Incompatible value with this classification task. Use a list of topics
        ValueError
            Some topic is not a string
        """

        if self.task == self.available_task[0]:
            result = self.__get_sentiment(X=X,
                                          name=name,
                                          split_num=split_num,
                                          delete_row_character=delete_row_character)
            logging.info("Sentiment analysis predicted")
        elif self.task == self.available_task[1]:
            result = self.__get_topic(X=X,
                                      name=name,
                                      split_num=split_num,
                                      labels_topic=labels_topic,
                                      delete_row_character=delete_row_character)
            logging.info("Topic predicted")
        return result

    def __get_sentiment(self,
                        delete_row_character,
                        X: pd.DataFrame,
                        name: str,
                        split_num: IntOrNone = None
                        ):
        """
        Get sentiment analysis of a tweet.

        Due to the processing time, the user is allowed to process to predict
        the sentiment of a collection of tweets
        chosen by the user (top x tweets - for testing).

        Parameters
        ----------
        X : pandas dataframe
            Original dataset
        name : str
            Column name
        split_num : IntOrNone
            Used to analyze the sentiment of a specific number
            of tweets (lower than the sample). Default is None
        delete_row_character : IntOrNone
            Delete row with number of characters less than a specific value by user.
            If value is None, no row will be deleted

        Returns
        -------
        X : pandas dataframe
            Output with the sentiment

        Raises
        ------
        ValueError
            Variable not available in the original dataframe
        TypeError
            Selected variable is not of type string
        ValueError
            If split_num is not None, its value must be greater than 0
        ValueError
          `delete_row_character` must be greater than 0
        """

        if name not in X.columns:
            raise ValueError("Variable not available in dataframe. "
                             f"Following variables are available in the data: {list(X.columns)}")

        if X[name].dtype != object:
            raise TypeError("Incorrect format. Column must be a `string` format")

        # delete rows with number of characters less than the user considered
        X = self.__delete_rows(X=X, name=name, delete_row_character=delete_row_character)

        X_serie = X[name]

        X_copy = X_serie.copy()

        # text data as list
        text_list = X_copy.astype("str").to_list()

        # fit the model according to user needs
        if split_num is None:
            classifier = self.__architecture_sentiment()
            sentiment_text = classifier(text_list)
            logging.info("Sentiment analysis predicted. All text")
        else:
            if split_num <= 0:
                logging.info("Error. Value must be greater than 0")
                raise ValueError("Value must be greater than 0")
            elif split_num < len(X_copy):
                logging.info("Sentiment analysis predicted. `split_num` text")
                classifier = self.__architecture_sentiment()
                sentiment_text = classifier(text_list[0: split_num])
            else:
                logging.info("Sentiment analysis predicted. All text")
                classifier = self.__architecture_sentiment()
                sentiment_text = classifier(text_list)

        sentiment_output = pd.DataFrame(sentiment_text)
        sentiment_output = sentiment_output.rename(columns={"label": "label_sentiment",
                                                            "score": "score_sentiment"})
        logging.info("Results about sentiment analysis")
        return sentiment_output

    def __architecture_sentiment(self):
        """
        Define the architecture of the model.
        Sentiment analysis task
        """
        # define the pretrained architecture
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        classifier = pipeline(self.task,
                              model=model,
                              tokenizer=tokenizer)
        logging.info("Define the architecture for sentiment analysis")
        return classifier

    def __get_topic(self,
                    delete_row_character,
                    X: pd.DataFrame,
                    name: str,
                    split_num: IntOrNone = None,
                    labels_topic: ListOrNone = None):
        """
        Get topic of a tweet.

        Due to the processing time, the user is allowed to process to predict
        the sentiment of a collection of tweets
        chosen by the user (top x tweets - for testing).

        Parameters
        ----------
        X : pandas dataframe
            Original dataset
        name : str
            Column name
        split_num : IntOrNone
            Used to analyze the sentiment of a specific number
            of tweets (lower than the sample). Default is None
        labels_topic : ListOrNone
            List with topic to analyse
        delete_row_character : IntOrNone
            Delete row with number of characters less than a specific value by user.
            If value is None, no row will be deleted

        Returns
        -------
        X : pandas dataframe
            Output with the topic and probabilities

        Raises
        ------
        ValueError
            Variable not available in the original dataframe
        TypeError
            Selected variable is not of type string
        ValueError
            If split_num is not None, its value must be greater than 0
         TypeError
            Incompatible value with this classification task. Use a list of topics
        ValueError
            Some topic is not a string
        """

        self.__check_errors_zs(labels_topic)

        if name not in X.columns:
            raise ValueError("Variable not available in dataframe. "
                             f"Following variables are available in the data: {list(X.columns)}")

        if X[name].dtype != object:
            raise TypeError("Incorrect format. Column must be a `string` format")

        # delete rows with number of characters less than the user considered
        X = self.__delete_rows(X=X, name=name, delete_row_character=delete_row_character)

        X_serie = X[name]

        X_copy = X_serie.copy()

        # text data as list
        text_list = X_copy.astype("str").to_list()

        # fit the model according to user needs
        if split_num is None:
            classifier = self.__architecture_topic()
            task_text = classifier(text_list, labels_topic, multi_label=True)
            logging.info("Topic predicted. All text")
        else:
            if split_num <= 0:
                logging.info("Error. Value must be greater than 0")
                raise ValueError("Value must be greater than 0")
            elif split_num < len(X_copy):
                logging.info("Topic predicted. `split_num` text")
                classifier = self.__architecture_topic()
                task_text = classifier(text_list[0: split_num],
                                       labels_topic,
                                       multi_label=True)
            else:
                logging.info("Topic predicted. All text")
                classifier = self.__architecture_topic()
                task_text = classifier(text_list, labels_topic, multi_label=True)

        output = pd.DataFrame()
        for i in task_text:
            df_base = pd.DataFrame(i)
            output = pd.concat([output, df_base], axis=0)
        return output

    def __check_errors_zs(self, labels):
        """
        Check correct format and values to classify the topic

        Raises
        ------
        TypeError
            Incompatible value with this classification task. Use a list of topics
        ValueError
            Some topic is not a string
        """

        if labels is None:
            raise TypeError("Incompatible value with this classification task. "
                            "Use a list of topics")

        check = all(isinstance(item, str) for item in labels)
        if not check:
            raise ValueError(f"Some topic is not a string: {labels}."
                             "Please, review the topics to be analyzed")

    def __architecture_topic(self):
        """
        Define the architecture of the model.
        Predict Topic task
        """
        # define the pretrained architecture
        classifier = pipeline(self.task,
                              model=self.model_name)
        logging.info("Define the architecture for topic prediction")
        return classifier

    @beartype
    def __delete_rows(self,
                      X: pd.DataFrame,
                      name: str,
                      delete_row_character: IntOrNone):
        """
        Delete row with number of characters less than a specific value by use

        Raises
        ------
        ValueError
          `delete_row_character` must be greater than 0
        """

        if isinstance(delete_row_character, int):
            if delete_row_character <= 0:
                raise ValueError("`delete_row_character` must be greater than 0")

        if delete_row_character is not None:
            X["length"] = X[name].str.len()
            X = X[X["length"] > delete_row_character]
        logging.info("Rows are deleted by user")
        return X.drop(["length"], axis=1)

