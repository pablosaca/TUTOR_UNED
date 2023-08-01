"""
######## PREPROCESSING ########

:Author: Pablo Sánchez Cabrera
:email: psancabrera@gmail.com

"""


import pandas as pd
import re
import string
import spacy
from beartype import beartype
import logging
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

logger = logging.getLogger(__name__)


class Preprocessing:
    @beartype
    def __init__(self,
                 mention: bool = True,
                 url: bool = True,
                 hashtags: bool = False,
                 lower: bool = True,
                 lemmatization: bool = False,
                 points: bool = True,
                 retweet: bool = True,
                 stop_words: bool = True,
                 truncate: bool = False,
                 n_truncate: int = 50):
        """
        Class to perform the pre-processing of the texts (tweets).

        The class design allows: \n
            - Removal of mentions \n
            - Removal of URLs \n
            - Removal of hashtags \n
            - Removal of retweets \n
            - Removal of punctuation marks and other characters (exclamations, question marks, etc). \n
            - Removal of stop-words \n
            - Use of lowercase \n
            - Lemmatization of the text \n

        The user is also provided with the ability to truncate the text.

        Parameters
        ----------
        mention : boolean
            If True, mentions are removed
        url : boolean
            If True, urls are removed
        hashtags : boolean
            If True, hashtags are removed
        lower : boolean
            If True, lowercase is applied
        lemmatization : boolean
            If True, lemmatization is applied
        points : boolean
            If True, punctuation marks are removed
        retweet : boolean
            If True, retweets are removed
        stop_words : boolean
            If True, removal stop-words in the text
        truncate : boolean
            If True, truncate is applied
        n_truncate : IntOrNone
            Number of characters in the text until the truncation is performed.
            Only used if truncate is True

        Raises
        ------
        ValueError
            If `truncate` is True, `n_truncate` must be greater than 0

        """
        self.mention = mention
        self.url = url
        self.hashtags = hashtags
        self.lower = lower
        self.lemmatization = lemmatization
        self.points = points
        self.retweet = retweet
        self.stop_words = stop_words
        self.truncate = truncate
        self.n_truncate = n_truncate

        if self.truncate:
            if self.n_truncate <= 0:
                raise ValueError("`n_truncate` must be greater than 0")

        logging.info("Define parameters for preprocessing process. Catch errors associated")

    @beartype
    def preprocessing_data(self,
                           X: pd.DataFrame,
                           name: str):
        """
        Preprocessing text

        Parameters
        ----------
        X : pandas dataframe
            Original dataset
        name : str
            Column name

        Returns
        -------
        X : pandas dataframe
            Output with the text column preprocessed

        Raises
        ------
        ValueError
            Variable not available in the original dataframe
        TypeError
            Selected variable is not of type string
        """

        if name not in X.columns:
            raise ValueError("Variable not available in dataframe. "
                             f"Following variables are available in the data: {list(X.columns)}")

        if X[name].dtype != object:
            raise TypeError("Incorrect format. Column must be a `string` format")

        X_serie = X[name]

        X_copy = X_serie.copy()

        X_copy = X_copy.apply(self.__prep_general)

        if self.mention:
            X_copy = X_copy.apply(self.__prep_mention)
        if self.url:
            X_copy = X_copy.apply(self.__prep_urls)
        if self.hashtags:
            X_copy = X_copy.apply(self.__prep_hashtags)
        if self.points:
            X_copy = X_copy.apply(self.__prep_points)
        if self.retweet:
            X_copy = X_copy.apply(self.__prep_retweet)
        if self.lower:
            X_copy = X_copy.apply(self.__prep_lower)
        if self.stop_words:
            X_copy = X_copy.apply(self.__prep_stopwords)
        if self.lemmatization:
            X_copy = X_copy.apply(self.__prep_lemmatization)

        X_copy = X_copy.apply(self.__prep_spaces)

        # truncate method is applied in the final processing
        if self.truncate:
            X_copy = X_copy.apply(self.__prep_truncate_text)

        X_copy = X_copy.apply(self.__prep_video)

        X_copy = X_copy.apply(self.__prep_spaces)

        logging.info("Preprocessing of the text made")

        return X_copy

    @beartype
    def select_columns(self,
                       X: pd.DataFrame,
                       columns_name: list):
        """
        Select columns more relevant for analysis

        Parameters
        ----------
        X : pandas dataframe
            Original dataset
        columns_name : list
            List with column names of dataframe

        Return
        ------
        X_copy : pandas dataframe
            Output with the selected variables

        Raises
        ------
        ValueError
            Variables not available in dataframe
        ValueError
            If stop_words is True, only available if `lower` and `points` is True
        ValueError
            If lemmatization is True, only available if `lower`,
            `points`, `mention` and `url` is True
        """

        X_copy = X.copy()

        columns = X.columns
        check = [i for i in columns_name if i in columns]

        if not check:
            logging.info("Error. Variables not available in dataframe")
            raise ValueError("Variables not available in dataframe. "
                             f"Please choose one of the following list: {list(X.columns)}")
        else:
            X_copy = X_copy[check]
            logging.info("New dataframe with columns selected by the user")
            return X_copy

    def __prep_mention(self, x):
        """
        Removal mentions

        Parameters
        ----------
        x : string

        Returns
        -------
        x : string
        """
        x = re.compile(r'@[\w_]+').sub('', x)
        return x

    def __prep_urls(self, x):
        """
        Removal URLs

        Parameters
        ----------
        x : string

        Returns
        -------
        x : string
        """
        x = re.compile(r'https?://[\w_./]+').sub('', x)
        return x

    def __prep_hashtags(self, x):
        """
        Removal hashtags

        Parameters
        ----------
        x : string

        Returns
        -------
        x : string
        """
        x = re.compile(r'#[\w_]+').sub('', x)
        return x

    def __prep_points(self, x):
        """
        Removal punctuation marks, exclamations, etc.

        Parameters
        ----------
        x : string

        Returns
        -------
        x : string
        """
        points = re.compile('[{}]'.format(re.escape(string.punctuation)))
        x = points.sub('', x)
        return x

    def __prep_retweet(self, x):
        """
        Removal retweet

        Parameters
        ----------
        x : string

        Returns
        -------
        x : string
        """
        x = re.compile('RT ').sub(' ', x)
        return x

    def __prep_lower(self, x):
        """
        Lowercase text

        Parameters
        ----------
        x : string

        Returns
        -------
        x : string
        """
        x = x.lower()
        return x

    def __prep_spaces(self, x):
        """
        Removal spaces

        Parameters
        ----------
        x : string

        Returns
        -------
        x : string
        """
        x = x.strip()
        return x

    def __prep_stopwords(self, x):
        """
        Delete stopwords

        The use of stop-words is conditional on the use of lowercase text.
        Punctuation marks and other signs (such as exclamations)
        have also been removed from said text.

        Parameters
        ----------
        x : string

        Returns
        -------
        x : string

        Raises
        ------
        ValueError
            Only available if `lower` and `points` is True
        """

        if (self.lower is False) or (self.points is False):
            raise ValueError("This method only works with lowercase "
                             "text and without punctuation marks. "
                             "Please, indicate `lower` and `points` as True")

        stop_words = set(stopwords.words('english'))
        nlp = spacy.load('en_core_web_md')

        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))

        tokens = nlp(x)
        tokens = [tok.lower_ for tok in tokens if not tok.is_punct
                  and not tok.is_space and not tok.is_digit]

        filtered_tokens = [pattern.sub('', token) for token in tokens if not (token in stop_words)]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def __prep_lemmatization(self, x):
        """
        Get lemmatization of words

        Only available if text is preprocessing according to next step:
        - Delete mentions
        - Delete url
        - Lowercase text
        - Delete he use of stop-words is conditional on the use of lowercase text.

        Parameters
        ----------
        x : string

        Returns
        -------
        x : string

        Raises
        ------
        ValueError
            Only available if `lower`,
            `points`, `mention` and `url` is True
        """

        cond1 = self.lower is False
        cond2 = self.points is False
        cond3 = self.mention is False
        cond4 = self.url is False

        if cond1 or cond2 or cond3 or cond4:
            raise ValueError("This method only works when next parameters are True"
                             " `lower`, `points`, `mention` and `url`")

        stop_words = set(stopwords.words('english'))
        nlp = spacy.load('en_core_web_md')

        pattern = re.compile(r'@[A-Za-z0-9_]+ | https?://[A-Za-z0-9./]+')
        pattern2 = re.compile('[{}]'.format(re.escape(string.punctuation)))

        text = pattern.sub('', x)
        tokens = nlp(text.lower())
        tokens = [tok.lemma_ for tok in tokens if not tok.is_punct
                  and not tok.is_space and not tok.is_digit]
        filtered_tokens = [pattern2.sub('', token) for token in tokens if not (token in stop_words)]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def __prep_general(self, x):
        """
        Delete general marks

        Parameters
        ----------
        x : string

        Returns
        -------
        x : string
        """

        x = re.compile('\r').sub(' ', x)
        x = re.compile('\n').sub(' ', x)
        x = re.compile('"').sub('', x)
        x = re.compile('[+-]').sub('', x)
        x = re.compile(r'<[\w_]+').sub('', x)
        x = re.compile(r'>').sub(' ', x)
        return x

    def __prep_truncate_text(self, x):
        """
        Truncate the text
        """
        return x[:self.n_truncate]

    def __prep_video(self, x):
        """
        Remove VIDEO word
        """
        x = re.compile(r'VIDEO | VIDEO').sub(' ', x)
        return x
