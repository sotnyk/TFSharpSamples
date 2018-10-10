using System;
using System.IO;
using System.Windows.Forms;

namespace ObjectDetectionSample
{
    static class Program
    {
        public static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        public static string WorkPath => Path.Combine(AppPath, "work");
        public static string DemoImagePath => Path.Combine(WorkPath, "input.jpg");
        public static string OutputImagePath => Path.Combine(WorkPath, "output.jpg");
        public static string LabelsPath => Path.Combine(WorkPath, "mscoco_label_map.pbtxt");
        public static string ModelPath => Path.Combine(WorkPath, "frozen_inference_graph.pb");


        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new ODMainForm());
        }
    }
}
