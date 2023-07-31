"""
######## CLASSIFICATION TASK UNIT TEST ########

:Author: Pablo Sánchez Cabrera
:email: psancabrera@gmail.com

"""

import pandas as pd
import unittest
from src.majorel.model.model_task import ClassificationTask

# import dataset
datos = pd.read_csv("../data/political_social_media.csv")


class TestSentimentAnalysis(unittest.TestCase):
    def test_check_task(self):
        """
        Check type of task
        """
        self.assertRaises(ValueError,
                          ClassificationTask,
                          task="classification")

    def test_check_model(self):
        """
        Check type of model
        """
        self.assertRaises(ValueError,
                          ClassificationTask,
                          model_name="model")

    def test_check_task_model(self):
        """
        Check type of model
        """
        self.assertRaises(ValueError,
                          ClassificationTask,
                          task="zero-shot-classification")

    def test_check_input_dataset(self):
        """
        Check columns is available in dataframe
        """
        sa = ClassificationTask()
        self.assertRaises(ValueError,
                          sa.get_task,
                          datos,
                          "pablo"
                          )

    def test_check_input_dataset2(self):
        """
        Check columns is available in dataframe
        """
        sa = ClassificationTask(task="zero-shot-classification",
                                model_name="facebook/bart-large-mnli")
        self.assertRaises(ValueError,
                          sa.get_task,
                          X=datos,
                          name="pablo",
                          labels_topic=["sport", "music"]
                          )

    def test_check_input_dataset3(self):
        """
        Check type of columns of dataframe
        """
        sa = ClassificationTask()
        self.assertRaises(TypeError,
                          sa.get_task,
                          datos,
                          "bias:confidence"
                          )

    def test_check_input_dataset4(self):
        """
        Check type of columns of dataframe
        """
        sa = ClassificationTask(task="zero-shot-classification",
                                model_name="facebook/bart-large-mnli")
        self.assertRaises(TypeError,
                          sa.get_task,
                          datos,
                          "bias:confidence"
                          )

    def test_check_n_tweets(self):
        """
        Check error when split_num is lower 0
        """
        n = 0
        sa = ClassificationTask()
        self.assertRaises(ValueError,
                          sa.get_task,
                          X=datos,
                          name="text",
                          split_num=n)

    def test_check_n_tweets2(self):
        """
        Check error when split_num is lower 0
        """
        n = -4
        sa = ClassificationTask(task="zero-shot-classification",
                                model_name="facebook/bart-large-mnli")
        self.assertRaises(ValueError,
                          sa.get_task,
                          X=datos,
                          name="text",
                          split_num=n,
                          labels_topic=["sport", "music"])

    def test_check_performance_sentiment(self):
        """
        Check suitable performance. Sentiment Analysis
        """
        n = 5
        columns_sa = 2
        sa = ClassificationTask()
        sentimentDF = sa.get_task(X=datos, name="text", split_num=n)

        self.assertIsInstance(sentimentDF, pd.DataFrame)
        self.assertEqual(len(sentimentDF), n)
        self.assertEqual(sentimentDF.shape[1], columns_sa)

    def test_check_topic(self):
        """
        Check not works if the topic label is wrong
        """
        n = 5
        sa = ClassificationTask(task="zero-shot-classification",
                                model_name="facebook/bart-large-mnli")

        self.assertRaises(TypeError,
                          sa.get_task,
                          X=datos,
                          name="text",
                          split_num=n,
                          labels_topic=None)

    def test_check_topic2(self):
        """
        Check not works if the topic label is wrong
        """
        n = 5
        labels = ["music", "sport", True]
        sa = ClassificationTask(task="zero-shot-classification",
                                model_name="facebook/bart-large-mnli")

        self.assertRaises(ValueError,
                          sa.get_task,
                          X=datos,
                          name="text",
                          split_num=n,
                          labels_topic=labels)

    def test_check_performance_topic(self):
        """
        Check suitable performance. Topic prediction
        """
        n = 5
        columns_topics = 3
        topics = ['health service', "education school"]
        sa = ClassificationTask(task="zero-shot-classification",
                                model_name="facebook/bart-large-mnli")

        # output provides the probability for each level
        topicDF = sa.get_task(X=datos, name="text", split_num=n, labels_topic=topics)

        self.assertIsInstance(topicDF, pd.DataFrame)
        self.assertEqual(len(topicDF), n*len(topics))
        self.assertEqual(topicDF.shape[1], columns_topics)

    def test_check_delete_rows(self):
        """
        Check error delete rows.
        Use to remove text with a number of characters lower than the user considered
        """
        n = 10
        rows = -5
        sa = ClassificationTask()
        self.assertRaises(ValueError,
                          sa.get_task,
                          X=datos,
                          name="text",
                          split_num=n,
                          delete_row_character=rows)
