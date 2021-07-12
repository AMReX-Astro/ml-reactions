import yt

def make_movie(file_list, save_dir='./', var='enuc', movie_name="flame.mp4"):
    i = 1
    for file in file_list:
        ds = yt.load(file)
        sl = yt.SlicePlot(ds,2,var)
        sl.save("movie_imag{}.png".format(str(i).zfill(4)))
        i+=1
    os.system("ffmpeg -r 60 -pattern_type glob -i 'movie_imag*.png' -vcodec mpeg4 -y {}".format(movie_name))
    os.system("rm movie_imag*")
    Video("movie.mp4", embed=True)
