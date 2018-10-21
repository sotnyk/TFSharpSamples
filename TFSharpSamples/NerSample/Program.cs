using CsvHelper;
using CsvHelper.Configuration;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using TensorFlow;

namespace NerSample
{
    class Program
    {
        public static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        public static string DataPath => Path.Combine(AppPath, "model");
        public static string ModelPath => Path.Combine(DataPath, "ner-bilstm.pb");
        public static string TokensPath => Path.Combine(DataPath, "tokens.csv");
        public static string TagsPath => Path.Combine(DataPath, "tags.csv");

        public static List<string> Index2Token = new List<string>();
        public static List<string> Index2Tag = new List<string>();
        public static Dictionary<string, int> Token2Index = new Dictionary<string, int>();
        public static Dictionary<string, int> Tag2Index = new Dictionary<string, int>();

        public static string[] samples = new[] {
            "Migrant caravan: 'May God soften Trump's heart'",
            "Thousands of mostly Honduran migrants are stuck at the Mexico-Guatemala border hoping to make it to the US.",
            "Saudi Arabia suggested on Friday Mr Khashoggi, a prominent Saudi critic, had died in a \"fist fight\".",
        };

        static void Main(string[] args)
        {
            // https://www.slideshare.net/trivadis/techevent-machine-learning - slide 32
            // Construct an in-memory graph from the serialized form.
            var graph = new TFGraph();
            // Load the serialized GraphDef from a file.
            var model = File.ReadAllBytes(ModelPath);
            (Index2Token, Token2Index) = ReadCsvDictionary(TokensPath);
            (Index2Tag, Tag2Index) = ReadCsvDictionary(TagsPath);

            graph.Import(model, "");
            using (var session = new TFSession(graph))
            {
                foreach(var sample in samples)
                {
                    var (tensor, lengths) = 
                        //CreateTensorFromSentence(sample);
                        CreateTestTensors();
                    var runner = session.GetRunner();
                    runner.AddInput(graph["input_batch"][0], tensor);
                    runner.AddInput(graph["lengths"][0], tensor);
                    runner.Fetch(graph["ArgMax"][0]);
                    //runner.Fetch("Softmax:0");
                    var output = runner.Run();
                    var result = output[0];
                    
                }
            }
        }

        private static (TFTensor, TFTensor) CreateTestTensors()
        {
            var x = new int[] {
                1061, 19693,  974,  980,  981,  811, 11474,  589,   39, 1058,   39,   57,
                1057,  4934,   46,    1,    1,    1,     1,    1,    1,    1,    1};
            var l = new int[] { 15 };
            var tensor = TFTensor.FromBuffer(new TFShape(1, x.Length), x, 0, x.Length);
            var lengths = TFTensor.FromBuffer(new TFShape(1), l, 0, l.Length);

            return (tensor, lengths);
        }

        private static (TFTensor, TFTensor) CreateTensorFromSentence(string sample)
        {
            var tokens = SentenceToTokens(sample);

            var featuresCount = 200;

            var features = new int[featuresCount];
            //new int[tokens.Count + 1];
            for (int tIndex = 0; tIndex < featuresCount; ++tIndex)
            {
                if (tIndex < tokens.Count())
                {
                    if (!Token2Index.TryGetValue(tokens[tIndex], out var oneHotIndex))
                        oneHotIndex = 0;
                    features[tIndex] = oneHotIndex;
                }
                else
                {
                    features[tIndex] = Token2Index["<PAD>"];
                }
            }
            // Add <PAD> token
            //features[tokens.Count] = Token2Index["<PAD>"];

            var tensor = TFTensor.FromBuffer(new TFShape(features.Length), features, 0, features.Length);
            var lengths = TFTensor.FromBuffer(new TFShape(1), new[] { tokens.Count }, 0, 1);

            /*
            var features = new int[(tokens.Count + 1) * Token2Index.Count];
            for (int tIndex = 0; tIndex<tokens.Count; ++tIndex)
            {
                if (!Token2Index.TryGetValue(tokens[tIndex], out var oneHotIndex))
                    oneHotIndex = 0;
                features[tIndex* Token2Index.Count + oneHotIndex] = 1;
            }
            // Add <PAD> token
            features[tokens.Count* Token2Index.Count+ Token2Index["<PAD>"]] = 1;

            var tensor = TFTensor.FromBuffer(new TFShape(1, tokens.Count + 1, Token2Index.Count), features, 0, features.Length);
            //var tensor = TFTensor.FromBuffer(new TFShape(tokens.Count + 1, Token2Index.Count), features, 0, features.Length);
            //var tensor = TFTensor.FromBuffer(new TFShape(Token2Index.Count, tokens.Count + 1), features, 0, features.Length);
            var lengths = TFTensor.FromBuffer(new TFShape(1), new[] { tokens.Count }, 0, 1);
            /*/

            /*const int MaxPlaceholderLength = 200;

            var features = new int[MaxPlaceholderLength * Token2Index.Count];
            var padIndex = Token2Index["<PAD>"];
            for (int tIndex = 0; tIndex < MaxPlaceholderLength; ++tIndex)
            {
                int oneHotIndex = padIndex;
                if (tIndex < tokens.Count)
                {
                    if (!Token2Index.TryGetValue(tokens[tIndex], out oneHotIndex))
                        oneHotIndex = 0;
                }
                features[tIndex * Token2Index.Count + oneHotIndex] = 1;
            }

            var tensor = TFTensor.FromBuffer(new TFShape(MaxPlaceholderLength, Token2Index.Count), features, 0, features.Length);
            //var tensor = TFTensor.FromBuffer(new TFShape(1, 1, Token2Index.Count, MaxPlaceholderLength), features, 0, features.Length);
            var lengths = //TFTensor.FromBuffer(new TFShape(1, 1), new[] { MaxPlaceholderLength }, 0, 1);
                    TFTensor.FromBuffer(new TFShape(1), new[] { MaxPlaceholderLength }, 0, 1);
            */

            return (tensor, lengths);
        }

        private static List<string> SentenceToTokens(string sentence)
        {
            var dirtyTokens = sentence.Split(new[] { ' ', '\r', '\n', '\t' }, StringSplitOptions.RemoveEmptyEntries);

            var tokens = new List<string>();
            foreach (var t in dirtyTokens)
            {
                var tLower = t.ToLowerInvariant();
                if (tLower.StartsWith("http://") || tLower.StartsWith("https://"))
                {
                    tokens.Add("<URL>");
                    continue;
                }
                if (t.StartsWith("@"))
                {
                    tokens.Add("nicknames");
                    continue;
                }
                foreach (var subtoken in DivideToken(t))
                    tokens.Add(subtoken);
            }

            /*foreach (var t in tokens)
                Console.WriteLine(t);
            Console.WriteLine();*/

            return tokens;
        }

        private static IEnumerable<string> DivideToken(string token)
        {
            var r = new Regex(@"("")|(\.)|(!)|(:)|(')|(,)", RegexOptions.Compiled);
            var result = new List<string>();
            var previoseWasPunctuation = false;
            foreach(var t in r.Split(token))
            {
                if (string.IsNullOrWhiteSpace(t))
                    continue;
                if (previoseWasPunctuation && !char.IsLetterOrDigit(t.First()))
                    result[result.Count - 1] += t;
                else
                    result.Add(t);
                previoseWasPunctuation = (t.Length == 1 && !char.IsLetterOrDigit(t[0]));
            }
            return result;
        }

        private static (List<string> tokensList, Dictionary<string, int> tokensDictionary) ReadCsvDictionary(string tokensPath)
        {
            var recordTypeDefinition = new
            {
                n = default(int),
                t = string.Empty,
            };
            using (var reader = File.OpenText(tokensPath))
            {
                var csv = new CsvReader(reader, new Configuration { Delimiter=","});
                var records = csv.GetRecords(recordTypeDefinition).ToList();
                var count = records.Max(r => r.n) + 1;
                var tokensList = new string[count].ToList();
                var tokensDictionary = new Dictionary<string, int>(count);
                foreach (var item in records)
                {
                    tokensList[item.n] = item.t;
                    tokensDictionary.Add(item.t, item.n);
                }
                return (tokensList, tokensDictionary);
            }            
        }
    }
}
