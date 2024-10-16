import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
##########################################################
# Byte Frequency Distribution
##########################################################


class ByteFrequencyDistribution():
    """Get byte frequency distribution table & shannon entropy results."""

    def get_fingerprint_by_folder(self, folder_path):
        """Get byte frequency distribution table & correlation matrix.
        @param foler_path: folder path.
        @return: fingerprint dict or None.
        """

        file_list = os.listdir(folder_path)
        file_list = [file for file in file_list if os.path.isfile(os.path.join(folder_path, file))]
        list_of_byte_frequency_table = []
        for file in file_list:
            byte_frequency_table, _ = self.get_byte_frequency_table_by_file_path(os.path.join(folder_path, file))
            if byte_frequency_table:
                list_of_byte_frequency_table.append(byte_frequency_table)
        
        average_byte_frequency_table = {k : sum(d[k] for d in list_of_byte_frequency_table) / len(list_of_byte_frequency_table) for k in list_of_byte_frequency_table[0].keys()}

        # transfomrm list of dict to pandas dataframe
        list_of_str_byte_frequency_table = [{str(k): v for k, v in d.items()} for d in list_of_byte_frequency_table]
        df = pd.DataFrame(list_of_str_byte_frequency_table)
        # filter out columns with 0 frequency
        df_new = df.loc[:, (df != 0).any(axis=0)]
        # calculate correlation matrix
        corr = df.corr()
        corr_new = df_new.corr()
        return corr, corr_new, average_byte_frequency_table


    def get_BFD_Corr_plots(self, data_type, input_folder_path, output_folder_path, threshold=0.7):
        """Get byte frequency distribution table & correlation matrix.
        @param data_type: data type, str..
        @param input_folder_path: input folder path.
        @param output_folder_path: output folder path.
        @return: None.
        """

        corr, corr_new, average_byte_frequency_table = self.get_fingerprint_by_folder(input_folder_path)
        corr.fillna(-0.5, inplace=True)
        ##################################################
        ############# plot correlation matrix
        ##################################################

        title_full = f'Full Correlation Matrix of Datatype [{data_type}]'
        title_truncated = f'Core Correlation Matrix of Datatype [{data_type}]'

        output_path_full = os.path.join(output_folder_path, f'{data_type}_corr_full.png')
        output_path_truncated = os.path.join(output_folder_path, f'{data_type}_corr_core.png')

        self.draw_corr(corr, title_full, threshold, output_path_full)
        self.draw_corr(corr_new, title_truncated, threshold, output_path_truncated)

        #########################################################
        ############# plot Byte Frequency Distribution matrix
        #######################################################

        title = f'Byte Frequency Distribution of Datatype [{data_type}]'
        output_path = os.path.join(output_folder_path, f'{data_type}_average_BFD.png')
        self.draw_BFD(average_byte_frequency_table, title, output_path)

    def draw_BFD(self, byte_frequency_table, title, output_path):
        """Draw byte frequency distribution table.
        @param byte_frequency_table: byte frequency table dict.
        @param output_path: output path.
        @return: None.
        """

        # plot byte frequency distribution
        plt.figure(figsize=(15, 12))
        sns.set_style("white")
        df = pd.DataFrame(list(byte_frequency_table.items()), columns=['Byte Value', 'Frequency'])
        sns.barplot(
                x='Byte Value',              # x-axis represents byte values (from 00 to FF)
                y='Frequency',         # y-axis represents their corresponding frequency in percentage
                data=df,               # data source is our DataFrame created from the frequency table
                width=1,
                color='blue',           # color of the bars
            )

        # Define the specific x-ticks you want to show (e.g., 1, 32, 64, etc.)
        byte_values_too_big = [i for i in range(256) if byte_frequency_table[str(format(i,'02X'))] > 0.4]
        xticks = list(range(0,256,32)) + [255]

        final_xticks = sorted(list(set(byte_values_too_big + xticks)))
        

        # Set the x-ticks as well as y-ticks to only show those values
        plt.xticks(ticks=final_xticks)
        plt.yticks(ticks=list(np.arange(0, 1.01, 0.1)))

        plt.title(title)
        plt.savefig(output_path)
        plt.close()


    def draw_corr(self, corr, title, threshold, output_path):

        plt.figure(figsize=(15, 12))
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        # all values > 0.5 
        result = [(column, index) for column in upper.columns for index in upper.index if upper[column][index] > threshold]

        # variable names
        variable_names = set([name for pair in result for name in pair])

        #  get all ticks positions
        all_ticks = np.arange(len(corr.columns))

        #  only keep the ticks of variables with correlation > 0.5
        filtered_ticks = [i for i, col in enumerate(corr.columns) if col in variable_names]

        # Only display labels greater than 0.5, and set other labels to empty strings
        xticks_labels = ['' if i not in filtered_ticks else corr.columns[i] for i in all_ticks]
        yticks_labels = ['' if i not in filtered_ticks else corr.index[i] for i in all_ticks]

        # only show half of the matrix
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask = mask, annot=False, fmt=".2f", cmap='coolwarm', linewidths=0.5)
        # Draw the heatmap with the mask and correct aspect ratio
        plt.title(title)
        plt.xticks(ticks=all_ticks, labels=xticks_labels, rotation=45)
        plt.yticks(ticks=all_ticks, labels=yticks_labels, rotation=45)

        plt.savefig(output_path)
        plt.close()

    def get_byte_frequency_table_by_byte_list(self, byte_stream):
        """Get byte frequency table dict.
        @param byte_stream: List[int], int value vaires from 0 to 255.
        @return: byte frequency table(over 0) dict or None.
        """

        byte_arr = byte_stream
        frequency_table = {}
        filesize = len(byte_arr)
        numerical_frequency_value = [0]*256
        for byte in byte_arr:
            numerical_frequency_value[byte] += 1
        max_frequency = max(numerical_frequency_value)
        numerical_frequency_value = [round(float(value),3) / max_frequency for value in numerical_frequency_value]
        numerical_frequency_table = {i: numerical_frequency_value[i] for i in range(256)}
        frequency_table = {str(format(i,'02X')): numerical_frequency_value[i] for i in range(256)}

        return frequency_table, numerical_frequency_table

    def get_byte_frequency_table_by_file_path(self, file_path):
        """Get byte frequency table dict.
        @param file_path: file path.
        @return: byte frequency table(over 0) dict or None.
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
        except IOError as e:
            print(e)
            return {}
        byte_arr = [byte for byte in data]

        frequency_table, numerical_frequency_table = self.get_byte_frequency_table_by_byte_list(byte_arr)
        return frequency_table, numerical_frequency_table

