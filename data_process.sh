mose_script="/home/qipeng/git/mosesdecoder/scripts"
token="$mose_script/tokenizer/tokenizer.perl"
lowercase="$mose_script/tokenizer/lowercase.perl"
cut_len="$mose_script/training/clean-corpus-n.perl"

python sgm2plain.py --in_fname="wmt_data/newstest2012-src.cs.sgm" --out_fname="data/news2012.cs"
python sgm2plain.py --in_fname="wmt_data/newstest2012-src.de.sgm" --out_fname="data/news2012.de"
python sgm2plain.py --in_fname="wmt_data/newstest2013-src.cs.sgm" --out_fname="data/news2013.cs"
python sgm2plain.py --in_fname="wmt_data/newstest2013-src.de.sgm" --out_fname="data/news2013.de"
python sgm2plain.py --in_fname="wmt_data/newstest2014-csen-src.cs.sgm" --out_fname="data/news2014.cs"
python sgm2plain.py --in_fname="wmt_data/newstest2014-deen-src.de.sgm" --out_fname="data/news2014.de"
python sgm2plain.py --in_fname="wmt_data/newstest2015-csen-src.cs.sgm" --out_fname="data/news2015.cs"
python sgm2plain.py --in_fname="wmt_data/newstest2015-deen-src.de.sgm" --out_fname="data/news2015.de"

perl $token -l cs -threads 8 < wmt_data/europarl-v7.cs-en.cs > data/eurov7.token.cs
perl $token -l de -threads 8 < wmt_data/europarl-v7.de-en.de > data/eurov7.token.de

perl $token -l cs -threads 4 < data/news2012.cs > data/news2012.token.cs
perl $token -l de -threads 4 < data/news2012.de > data/news2012.token.de
perl $token -l cs -threads 4 < data/news2013.cs > data/news2013.token.cs
perl $token -l de -threads 4 < data/news2013.de > data/news2013.token.de
perl $token -l cs -threads 4 < data/news2014.cs > data/news2014.token.cs
perl $token -l de -threads 4 < data/news2014.de > data/news2014.token.de
perl $token -l cs -threads 4 < data/news2015.cs > data/news2015.token.cs
perl $token -l de -threads 4 < data/news2015.de > data/news2015.token.de

perl $lowercase < data/eurov7.token.cs > data/eurov7.cs
perl $lowercase < data/eurov7.token.de > data/eurov7.de
perl $lowercase < data/news2012.token.cs > data/news2012.cs
perl $lowercase < data/news2012.token.de > data/news2012.de
perl $lowercase < data/news2013.token.cs > data/news2013.cs
perl $lowercase < data/news2013.token.de > data/news2013.de
perl $lowercase < data/news2014.token.cs > data/news2014.cs
perl $lowercase < data/news2014.token.de > data/news2014.de
perl $lowercase < data/news2015.token.cs > data/news2015.cs
perl $lowercase < data/news2015.token.de > data/news2015.de

perl $cut_len data/eurov7 cs de data/eurov7.clean 1 80
perl $cut_len data/news2012 cs de data/news2012.clean 1 80
perl $cut_len data/news2013 cs de data/news2013.clean 1 80
perl $cut_len data/news2014 cs de data/news2014.clean 1 80
perl $cut_len data/news2015 cs de data/news2015.clean 1 80

mv data/eurov7.clean.cs data/cs.train
cat data/news2012.clean.cs data/news2013.clean.cs > data/cs.valid
cat data/news2014.clean.cs data/news2015.clean.cs > data/cs.test

mv data/eurov7.clean.de data/de.train
cat data/news2012.clean.de data/news2013.clean.de > data/de.valid
cat data/news2014.clean.de data/news2015.clean.de > data/de.test



