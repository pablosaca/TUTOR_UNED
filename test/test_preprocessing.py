"""
######## PREPROCESSING UNIT TEST ########

:Author: Pablo Sánchez Cabrera
:email: psancabrera@gmail.com

"""

import pandas as pd
import unittest
import spacy
from nltk.corpus import stopwords
import nltk
from src.majorel.preprocessing import Preprocessing
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_md')

# import dataset
datos = pd.read_csv("../data/political_social_media.csv")


class TestPreprocessing(unittest.TestCase):

    def test_preprocessing_truncate(self):
        self.assertRaises(ValueError,
                          Preprocessing,
                          truncate=True,
                          n_truncate=-5)

    def test_check_preprocessing_input(self):
        """
        Check columns is available in dataframe
        """
        text_prep = Preprocessing()
        self.assertRaises(ValueError,
                          text_prep.preprocessing_data,
                          datos,
                          "pablo"
                          )

    def test_check_preprocessing_input2(self):
        """
        Check type of columns of dataframe
        """
        text_prep = Preprocessing()
        self.assertRaises(TypeError,
                          text_prep.preprocessing_data,
                          datos,
                          "bias:confidence"
                          )

    def test_preprocessing_stopwords(self):
        """
        Check use stop_words process when conditions are available
        """
        text_prep = Preprocessing(mention=True,
                                  url=True,
                                  hashtags=False,
                                  retweet=True,
                                  lower=False,
                                  lemmatization=False,
                                  points=True,
                                  stop_words=True)
        self.assertRaises(ValueError,
                          text_prep.preprocessing_data,
                          datos,
                          "text")

    def test_preprocessing_lemma(self):
        """
        Check use lemmatization process when conditions are available
        """
        text_prep = Preprocessing(mention=False,
                                  url=True,
                                  hashtags=False,
                                  retweet=True,
                                  lower=False,
                                  lemmatization=True,
                                  points=True,
                                  stop_words=True)
        self.assertRaises(ValueError,
                          text_prep.preprocessing_data,
                          datos,
                          "text")

    def test_check_select_columns(self):
        """
        Check select required columns
        """
        list_columns = ["pablo", "sanchez"]
        text_prep = Preprocessing()
        self.assertRaises(ValueError,
                          text_prep.select_columns,
                          datos,
                          list_columns)

    def test_check_select_columns1(self):
        """
        Check select required columns. It is verified that:
        - Output is a dataframe
        - Output has two columns
        - Output has the same rows as the original file

        """
        list_columns = ["_unit_id", "source"]
        text_prep = Preprocessing()
        datos_select = text_prep.select_columns(datos, list_columns)
        self.assertIsInstance(datos_select, pd.DataFrame)
        self.assertEqual(datos_select.shape[1], len(list_columns))
        self.assertEqual(datos_select.shape[0], datos.shape[0])

    def test_check_select_columns2(self):
        """
        Check select required columns. It is verified that:
        - Output is a dataframe
        - Output has two columns
        - Output has the same rows as the original file

        Note: excluding variables that do not exist in the original file
        """

        list_columns = ["_unit_id", "source", "pablo"]
        text_prep = Preprocessing()
        datos_select = text_prep.select_columns(datos, list_columns)
        self.assertIsInstance(datos_select, pd.DataFrame)
        self.assertEqual(datos_select.shape[1], len(list_columns) - 1)
        self.assertEqual(datos_select.shape[0], datos.shape[0])

    def test_check_good_dev(self):
        """
        Check correct performance. Preprocessing data.
        - Select columns
        - Simple preprocessing

        It is verified that:
        - Output is a dataframe
        - Output has list_columns + 1
        - Output has the same rows as the original file

        """
        list_columns = ["_unit_id", "source", "message", "text", "label"]
        text_prep = Preprocessing(mention=True,
                                  url=True,
                                  hashtags=False,
                                  retweet=True,
                                  lower=True,
                                  lemmatization=False,
                                  points=True,
                                  stop_words=False)
        datos_select = text_prep.select_columns(datos, columns_name=list_columns)
        datos_select["processing_text"] = text_prep.preprocessing_data(datos_select, "text")

        self.assertIsInstance(datos_select, pd.DataFrame)
        self.assertEqual(datos_select.shape[1], len(list_columns) + 1)
        self.assertEqual(datos_select.shape[0], datos.shape[0])
