import unittest
from unittest.mock import patch

import without_camera as wc


class ChooseAndDetectTest(unittest.TestCase):
    @patch("without_camera.classify_tomato")
    @patch("without_camera.pilih_gambar", return_value="tomat1.jpg")
    def test_choose_and_detect_runs_detection_when_image_selected(
        self, mock_pilih_gambar, mock_classify_tomato
    ):
        wc.choose_and_detect()

        mock_pilih_gambar.assert_called_once_with()
        mock_classify_tomato.assert_called_once_with("tomat1.jpg")

    @patch("without_camera.messagebox.showinfo")
    @patch("without_camera.classify_tomato")
    @patch("without_camera.pilih_gambar", return_value="")
    def test_choose_and_detect_shows_cancel_when_no_image_selected(
        self, mock_pilih_gambar, mock_classify_tomato, mock_showinfo
    ):
        wc.choose_and_detect()

        mock_pilih_gambar.assert_called_once_with()
        mock_classify_tomato.assert_not_called()
        mock_showinfo.assert_called_once_with("Batal", "Tidak ada file yang dipilih.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
