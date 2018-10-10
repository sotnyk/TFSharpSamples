using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using System.Windows.Forms;
using TensorFlow;
using TFSharpSamples.Common;

namespace ObjectDetectionSample
{
    public partial class ODMainForm : Form
    {
        TFGraph _graph;
        IEnumerable<CatalogItem> _catalog;
        private TFSession _session;

        private static double MIN_SCORE_FOR_OBJECT_HIGHLIGHTING = 0.5;

        public ODMainForm()
        {
            InitializeComponent();
        }

        private async void ODMainForm_Load(object sender, EventArgs e)
        {
            try
            {
                toolStripStatusLabel.Text = "Start to download model and labels";
                await DownloadDefaultModelAsync(Program.ModelPath);
                await DownloadDefaultTexts(Program.LabelsPath);
                toolStripStatusLabel.Text = "Start to load model.";
                _catalog = CatalogUtil.ReadCatalogItems(Program.LabelsPath);
                _graph = new TFGraph();
                var model = File.ReadAllBytes(Program.ModelPath);
                _graph.Import(new TFBuffer(model));
                _session = new TFSession(_graph);
                toolStripStatusLabel.Text = "Initialization done.";
                await DrawBoxesAsync(Program.DemoImagePath);
                loadPictureToolStripMenuItem.Enabled = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString(), "Error during downloading model and labels",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
                toolStripStatusLabel.Text = "Error during downloading model and labels";
            }
        }

        private async Task DownloadDefaultModelAsync(string modelPath)
        {
            string defaultModelUrl = ConfigurationManager.AppSettings["DefaultModelUrl"] ?? throw new ConfigurationErrorsException("'DefaultModelUrl' setting is missing in the configuration file");

            var dir = Path.GetDirectoryName(modelPath);
            var modelShortName = Path.GetFileName(modelPath);
            var zipfile = Path.Combine(dir, "faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz");

            if (File.Exists(modelPath))
                return ;

            if (!File.Exists(zipfile))
            {
                toolStripStatusLabel.Text = "Downloading default model";
                var wc = new HttpClient();
                var content = await wc.GetByteArrayAsync(new Uri(defaultModelUrl));
                File.WriteAllBytes(zipfile, content);
            }

            ExtractToDirectory(zipfile, dir);
            File.Delete(zipfile);
            var tempModelName = Directory.GetFiles(dir, modelShortName, SearchOption.AllDirectories).FirstOrDefault(f => f != modelPath);
            if (tempModelName == null)
                throw new IOException($"Cannot find unpacked model file '{modelShortName}'");
            File.Copy(tempModelName, modelPath, true);

            var unpackedArchiveDir = Path.GetDirectoryName(tempModelName);
            foreach (var f in Directory.GetFiles(unpackedArchiveDir))
                File.Delete(f);
            Directory.Delete(unpackedArchiveDir);
        }

        private static void ExtractToDirectory(string file, string targetDir)
        {
            Console.WriteLine("Extracting");

            using (Stream inStream = File.OpenRead(file))
            using (Stream gzipStream = new GZipInputStream(inStream))
            {
                TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream);
                tarArchive.ExtractContents(targetDir);
            }
        }

        private async Task DownloadDefaultTexts(string labelsPath)
        {
            toolStripStatusLabel.Text = "Downloading default label map";

            string defaultTextsUrl = ConfigurationManager.AppSettings["DefaultTextsUrl"] ?? throw new ConfigurationErrorsException("'DefaultTextsUrl' setting is missing in the configuration file");
            var wc = new HttpClient();
            var content = await wc.GetByteArrayAsync(defaultTextsUrl);
            File.WriteAllBytes(labelsPath, content);
        }

        private async void loadPictureToolStripMenuItem_Click(object sender, EventArgs e)
        {
            loadPictureToolStripMenuItem.Enabled = false;
            try
            {
                if (openFileDialog.ShowDialog(this) == DialogResult.OK)
                {
                    await DrawBoxesAsync(openFileDialog.FileName);
                }
            }
            finally
            {
                loadPictureToolStripMenuItem.Enabled = true;
            }
        }

        private async Task DrawBoxesAsync(string inputFileName)
        {
            toolStripStatusLabel.Text = $"Start processing of '{inputFileName}'";
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            var drawBoxesTask = new Task(() => DrawBoxes(inputFileName));
            drawBoxesTask.Start();
            await drawBoxesTask;
            using (var imageStream = File.OpenRead(Program.OutputImagePath))
                pbOutput.Image = Image.FromStream(imageStream);
            stopwatch.Stop();
            toolStripStatusLabel.Text = $"Processing of '{inputFileName}' done in {stopwatch.ElapsedMilliseconds} msec.";
        }

        private void DrawBoxes(string inputFileName)
        {
            var tensor = ImageUtil.CreateTensorFromImageFile(inputFileName, TFDataType.UInt8);
            var runner = _session.GetRunner();

            runner
                .AddInput(_graph["image_tensor"][0], tensor)
                .Fetch(
                _graph["detection_boxes"][0],
                _graph["detection_scores"][0],
                _graph["detection_classes"][0],
                _graph["num_detections"][0]);
            var output = runner.Run();

            var boxes = (float[,,])output[0].GetValue(jagged: false);
            var scores = (float[,])output[1].GetValue(jagged: false);
            var classes = (float[,])output[2].GetValue(jagged: false);
            var num = (float[])output[3].GetValue(jagged: false);

            DrawBoxes(boxes, scores, classes, inputFileName, Program.OutputImagePath, MIN_SCORE_FOR_OBJECT_HIGHLIGHTING);
        }

        private void DrawBoxes(float[,,] boxes, float[,] scores, float[,] classes, string inputFile, string outputFile, double minScore)
        {
            var x = boxes.GetLength(0);
            var y = boxes.GetLength(1);
            var z = boxes.GetLength(2);

            float ymin = 0, xmin = 0, ymax = 0, xmax = 0;

            using (var editor = new ImageEditor(inputFile, outputFile))
            {
                for (int i = 0; i < x; i++)
                {
                    for (int j = 0; j < y; j++)
                    {
                        if (scores[i, j] < minScore) continue;

                        for (int k = 0; k < z; k++)
                        {
                            var box = boxes[i, j, k];
                            switch (k)
                            {
                                case 0:
                                    ymin = box;
                                    break;
                                case 1:
                                    xmin = box;
                                    break;
                                case 2:
                                    ymax = box;
                                    break;
                                case 3:
                                    xmax = box;
                                    break;
                            }
                        }

                        int value = Convert.ToInt32(classes[i, j]);
                        CatalogItem catalogItem = _catalog.FirstOrDefault(item => item.Id == value);
                        editor.AddBox(xmin, xmax, ymin, ymax, $"{catalogItem.DisplayName} : {(scores[i, j] * 100).ToString("0")}%");
                    }
                }
            }
        }
    }
}
