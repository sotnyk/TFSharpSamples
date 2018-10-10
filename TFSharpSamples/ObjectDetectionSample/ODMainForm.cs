using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Configuration;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ObjectDetectionSample
{
    public partial class ODMainForm : Form
    {
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
                toolStripStatusLabel.Text = "Start to download model and labels";
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
            Directory.Delete(Path.GetDirectoryName(tempModelName));
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
    }
}
