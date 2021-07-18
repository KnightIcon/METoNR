import abc


class BaseIterator(object):
    @abc.abstractmethod
    def parser_one_line(self, line):
        pass

    @abc.abstractmethod
    def load_data_from_file(self, infile):
        pass

    @abc.abstractmethod
    def _convert_data(self, labels, features):
        pass

    @abc.abstractmethod
    def gen_feed_dict(self, data_dict):
        pass
