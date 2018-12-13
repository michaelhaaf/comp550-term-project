import re

class DocumentFactory:

    TAGS_PER_DOC = 1000

    def create_documents(self, book_data_dict):
        documents = []
        for book_file_path, tag_sequence in book_data_dict.items():
            label = self.get_label_from_file_path(book_file_path)
            documents.extend(self.divide_book_into_documents(label, tag_sequence)[:-1])
        return documents

    def divide_book_into_documents(self, label, tag_sequence):
        return [Document(label, tag_sequence[index: index + self.TAGS_PER_DOC])
                for index in range(0, len(tag_sequence), self.TAGS_PER_DOC)]

    def get_label_from_file_path(self, file_path):
        label = self.get_author_name_from_path_regex(file_path)
        if label is None:
            return file_path
        else:
            return label.group()

    def get_author_name_from_path_regex(self, file_path):
        return re.search('(?<=-)(.+)(?=-)', file_path)


class Document:

    def __init__(self, label, tag_sequence):
        self.label = label
        self.tag_sequence = tag_sequence


