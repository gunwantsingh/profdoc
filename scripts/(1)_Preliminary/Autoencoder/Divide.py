import os
import pandas as pd
import uuid


class FileSettings(object):
    def __init__(self, file_name, row_size=100):
        self.file_name = file_name
        self.row_size = row_size


class FileSplitter(object):

    def __init__(self, file_settings):
        self.file_settings = file_settings

        if type(self.file_settings).__name__ != "FileSettings":
            raise Exception("Please pass correct instance ")

        self.df = pd.read_csv(self.file_settings.file_name,
                              chunksize=self.file_settings.row_size)

    def run(self, directory="temp"):

        try:os.makedirs(directory)
        except Exception as e:pass

        counter = 0

        while True:
            try:
                file_name = "{}/{}_{}_row_{}_{}.csv".format(
                    directory,  self.file_settings.file_name.split(".")[0], counter, self.file_settings.row_size, uuid.uuid4().__str__()
                )
                df = next(self.df).to_csv(file_name)
                counter = counter + 1
            except StopIteration:
                break
            except Exception as e:
                print("Error:",e)
                break

        return True


def main():
    helper =  FileSplitter(FileSettings(
        file_name='UNSW_NB15_training-set.csv',
        row_size=41166
    ))
    helper.run()

main()