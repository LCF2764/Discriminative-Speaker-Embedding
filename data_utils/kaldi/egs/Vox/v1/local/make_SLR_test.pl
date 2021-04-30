#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#
# Usage: make_SLR_test.pl /export/voxceleb2 dev data/dev
#
# Note: This script requires ffmpeg to be installed and its location included in $PATH.

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-SLR-testsets> <path-to-data-dir>\n";
  print STDERR "e.g. $0 $data/test data/test\n";
  exit(1);
}

($data_base, $out_dir) = @ARGV;

opendir my $dh, "$data_base" or die "Cannot open directory: $!";
my @files = map{s/\.[^.]+$//;$_}grep {/\.wav$/} readdir($dh);
#my @spkr_dirs = grep {-d "$data_base/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

if (system("mkdir -p $out_dir") != 0) {
  die "Error making directory $out_dir";
}

open(SPKR, ">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

foreach (@files) {
  my $name = $_;
  #my $wav = "ffmpeg -v 8 -i $data_base/$dataset/aac/$spkr_id/$rec_id/$name.m4a -f wav -acodec pcm_s16le -|";
  my $wav = "$data_base/$name.wav";
  my $utt_id = "$name";
  print WAV "$utt_id", " $wav", "\n";
  print SPKR "$utt_id", " $utt_id", "\n";

}
close(SPKR) or die;
close(WAV) or die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
