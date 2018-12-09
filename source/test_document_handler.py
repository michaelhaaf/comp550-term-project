from unittest import TestCase
from document_handler import DocumentFactory

class TestDocumentHandler(TestCase):

    def test_create_documents(self):
        any_tag = ('token', 'pos', 'lemma')
        file_path_one = 'year-author1-title.txt'
        file_path_two = 'year-author2-title.txt'
        two_doc_tag_sequence = [any_tag] * DocumentFactory().TAGS_PER_DOC * 2
        three_and_half_doc_tag_sequence = [any_tag] * int(DocumentFactory.TAGS_PER_DOC * 3.5)

        result = DocumentFactory().create_documents({file_path_one: two_doc_tag_sequence,
                                                     file_path_two: three_and_half_doc_tag_sequence})

        self.assertEqual('author1', result[0].label)
        self.assertEqual('author2', result[1].label)    # we always drop the last document, so next doc is author2
        self.assertEqual('author2', result[3].label)
        self.assertEqual(4, len(result))    # (2-1 + 3.5-0.5 = 4)


    def test_get_label_from_file_path(self):
        any_file_path = '1929-Jean Giono-Un de Baumugnes.txt'
        result = DocumentFactory().get_label_from_file_path(any_file_path)
        self.assertEqual('Jean Giono', result)