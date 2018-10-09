using Mono.Options;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using TensorFlow;
using TFSharpSamples.Common;

namespace InceptionSample
{
    class Program
    {
        static void Error(string msg)
        {
            Console.WriteLine("Error: {0}", msg);
            Environment.Exit(1);
        }

        static void Help()
        {
            options.WriteOptionDescriptions(Console.Out);
        }

        static bool jagged = true;

        static OptionSet options = new OptionSet()
        {
            { "m|dir=",  "Specifies the directory where the model and labels are stored", v => dir = v },
            { "h|help", v => Help () },

            { "amulti", "Use multi-dimensional arrays instead of jagged arrays", v => jagged = false }
        };


        static string dir, modelFile, labelsFile;
        public static void Main(string[] args)
        {
            var files = options.Parse(args);
            if (dir == null)
            {
                dir = "work";
                //Error ("Must specify a directory with -m to store the training data");
            }

            //if (files == null || files.Count == 0)
            //	Error ("No files were specified");

            if (files.Count == 0)
            {
                //files = new List<string>() { "work\\parrot.jpg" };
                files = Directory.GetFiles("work", "*.jpg").ToList();
            }

            ModelFiles(dir);

            // Construct an in-memory graph from the serialized form.
            var graph = new TFGraph();
            // Load the serialized GraphDef from a file.
            var model = File.ReadAllBytes(modelFile);

            graph.Import(model, "");
            using (var session = new TFSession(graph))
            {
                var labels = File.ReadAllLines(labelsFile);

                foreach (var file in files)
                {
                    Console.WriteLine($"Process file: '{file}'");
                    // Run inference on the image files
                    // For multiple images, session.Run() can be called in a loop (and
                    // concurrently). Alternatively, images can be batched since the model
                    // accepts batches of image data as input.
                    var tensor = ImageUtil.CreateTensorFromImageFile(file);

                    var runner = session.GetRunner();
                    runner.AddInput(graph["input"][0], tensor).Fetch(graph["output"][0]);
                    var output = runner.Run();
                    // output[0].Value() is a vector containing probabilities of
                    // labels for each image in the "batch". The batch size was 1.
                    // Find the most probably label index.

                    var result = output[0];
                    var rshape = result.Shape;
                    if (result.NumDims != 2 || rshape[0] != 1)
                    {
                        var shape = "";
                        foreach (var d in rshape)
                        {
                            shape += $"{d} ";
                        }
                        shape = shape.Trim();
                        Console.WriteLine($"Error: expected to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape [{shape}]");
                        Environment.Exit(1);
                    }

                    // You can get the data in two ways, as a multi-dimensional array, or arrays of arrays, 
                    // code can be nicer to read with one or the other, pick it based on how you want to process
                    // it
                    bool jagged = true;

                    var bestIdx = 0;
                    float p = 0, best = 0;

                    if (jagged)
                    {
                        var probabilities = ((float[][])result.GetValue(jagged: true))[0];
                        for (int i = 0; i < probabilities.Length; i++)
                        {
                            if (probabilities[i] > best)
                            {
                                bestIdx = i;
                                best = probabilities[i];
                            }
                        }

                    }
                    else
                    {
                        var val = (float[,])result.GetValue(jagged: false);

                        // Result is [1,N], flatten array
                        for (int i = 0; i < val.GetLength(1); i++)
                        {
                            if (val[0, i] > best)
                            {
                                bestIdx = i;
                                best = val[0, i];
                            }
                        }
                    }

                    Console.WriteLine($"{file} best match: [{bestIdx}] {best * 100.0}% {labels[bestIdx]}");
                }
            }
#if DEBUG
            Console.WriteLine("Press <Enter> to exit.");
            Console.ReadLine();
#endif
        }

        //
        // Downloads the inception graph and labels
        //
        static void ModelFiles(string dir)
        {
            string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";

            modelFile = Path.Combine(dir, "tensorflow_inception_graph.pb");
            labelsFile = Path.Combine(dir, "imagenet_comp_graph_label_strings.txt");
            var zipfile = Path.Combine(dir, "inception5h.zip");

            if (File.Exists(modelFile) && File.Exists(labelsFile))
                return;

            Directory.CreateDirectory(dir);
            var wc = new WebClient();
            wc.DownloadFile(url, zipfile);
            ZipFile.ExtractToDirectory(zipfile, dir);
            File.Delete(zipfile);
        }
    }
}
