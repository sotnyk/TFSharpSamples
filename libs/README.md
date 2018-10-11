Prebuilt binaries

Was found in https://stackoverflow.com/questions/44682845/tensorflow-c-library-available-for-windows: in Darbean's answer (https://stackoverflow.com/a/52414501/4884761)

To find actual binaries, transform the original link https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.9.0.zip:
* Change the version number to your actual Tensorflow version number (for me it is 1.11.0)
* Change "cpu" to "gpu" if you want to find CUDA based dll.

You can copy selected DLL into your debug/release folder and all should works.

After some tests I found that CPU version is slow but works on any Windows x64 machine. GPU version works on machines where is installed Tensorflow GPU version (with proper hardware, of couse). Maybe it can work without full installation, but TF has detailed instructions how to install environment and to check it. So, it was simpler way for me.
