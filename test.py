from pytube import YouTube
from tqdm import tqdm

with open("link.txt", 'r') as file:
    lines = file.readlines()
    for line in (lines):
        line = line.strip()
        try:
            while True:
                try:
                    source = YouTube(line)
                    vid = source.streams.first()
                    break
                except:
                    pass

            en_caption = source.captions.get_by_language_code('en')

            en_caption_convert_to_srt =(en_caption.generate_srt_captions())
            
            text_file = open(f"./sub/{vid.title[:-4]}.txt", "w")
            text_file.write(en_caption_convert_to_srt)
            text_file.close()
        except:
            print(line)

# source = YouTube('https://www.youtube.com/watch?v=wjTn_EkgQRg&index=1&list=PLgJ7b1NurjD2oN5ZXbKbPjuI04d_S0V1K')

# en_caption = source.captions.get_by_language_code('en')

# en_caption_convert_to_srt =(en_caption.generate_srt_captions())

# print(en_caption_convert_to_srt)
# #save the caption to a file named Output.txt
# text_file = open("Output.txt", "w")
# text_file.write(en_caption_convert_to_srt)
# text_file.close()