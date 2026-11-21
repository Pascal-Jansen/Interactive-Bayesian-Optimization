using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using BayesianOptimization.Examples;
using BayesianOptimization.Scripts;
using NUnit.Framework;
using UnityEngine;

namespace BayesianOptimization.Tests.EditMode
{
    public class IntegrationConfigurationEditModeTests
    {
        [TestCase(0)]
        [TestCase(-1)]
        [TestCase(65536)]
        public void RefusesInvalidLoopbackPorts(int port)
        {
            var target = new GameObject("PortValidationTest");
            try
            {
                var manager = target.AddComponent<BoForUnityManager>();
                manager.port = port;
                Assert.That(string.Join(" ", BoConfigValidator.GetErrors(manager)), Does.Contain("TCP port"));
            }
            finally { UnityEngine.Object.DestroyImmediate(target); }
        }

        [TestCase(1)]
        [TestCase(56001)]
        [TestCase(65535)]
        public void AcceptsValidLoopbackPorts(int port)
        {
            var target = new GameObject("PortValidationTest");
            try
            {
                var manager = target.AddComponent<BoForUnityManager>();
                manager.port = port;
                Assert.That(string.Join(" ", BoConfigValidator.GetErrors(manager)), Does.Not.Contain("TCP port"));
            }
            finally { UnityEngine.Object.DestroyImmediate(target); }
        }

        [Test]
        public void PassesTheConfiguredPortToPython()
        {
            var start = new ProcessStartInfo();
            typeof(PythonStarter).GetMethod("ConfigurePythonRuntimeEnvironment", BindingFlags.Static | BindingFlags.NonPublic)
                .Invoke(null, new object[] { start, "test-log-root", false, "botorch", 56002 });
            Assert.That(start.Environment["BO_PORT"], Is.EqualTo("56002"));
        }

        [Test]
        public void ProtocolDtoStillSerializesDictionaryAndNullableIteration()
        {
            Type dtoType = typeof(SocketNetwork).Assembly.GetType("BayesianOptimization.Scripts.ParametersMsg");
            object dto = Activator.CreateInstance(dtoType);
            dtoType.GetField("type").SetValue(dto, "parameters");
            dtoType.GetField("values").SetValue(dto, new Dictionary<string, float> { { "x", 0.25f } });
            dtoType.GetField("iteration").SetValue(dto, (int?)7);
            string json = (string)typeof(SocketNetwork).GetMethod("SerializeJson", BindingFlags.Static | BindingFlags.NonPublic)
                .Invoke(null, new[] { dto });
            Assert.That(json, Does.Contain("\"values\":{\"x\":0.25}"));
            Assert.That(json, Does.Contain("\"iteration\":7"));
        }

        [Test]
        public void AnalyticObjectiveHasTheDocumentedMinimumAndFiniteRange()
        {
            Assert.That(AnalyticOptimizationExample.Evaluate(0.25f, 0.75f), Is.Zero);
            Assert.That(AnalyticOptimizationExample.Evaluate(1f, 0f), Is.EqualTo(1.125f));
        }

        [Test]
        public void ValidatesHierarchicalParameterGatingAndKernelPresetNormalization()
        {
            var target = new GameObject("GateValidationTest");
            try
            {
                var manager = target.AddComponent<BoForUnityManager>();
                manager.optimizerBackend = BoForUnityManager.OptimizerBackend.BoTorch;
                manager.botorchKernelPreset = BoForUnityManager.BoTorchKernelPreset.SharedAddInterMatern32;
                manager.parameters = new List<ParameterEntry>
                {
                    new ParameterEntry("alpha", new ParameterArgs(0f, 1f)),
                    new ParameterEntry("size", new ParameterArgs(0.5f, 2f)
                    {
                        gatedBy = "alpha",
                        gateThreshold = 0.05f,
                        gateNeutral = 0f,
                        gateSharpness = 20f
                    }),
                    new ParameterEntry("hue", new ParameterArgs(0f, 360f)
                    {
                        gatedBy = "alpha",
                        gateThreshold = 0f,
                        gateNeutral = 0.5f
                    })
                };
                manager.objectives = new List<ObjectiveEntry>
                {
                    new ObjectiveEntry("trust", new ObjectiveArgs(1f, 7f, false, 1))
                };

                Assert.That(BoConfigValidator.GetErrors(manager), Is.Empty);

                var normalizeMethod = typeof(SocketNetwork).GetMethod(
                    "NormalizeKernelPreset",
                    BindingFlags.Static | BindingFlags.NonPublic
                );
                Assert.That(
                    normalizeMethod.Invoke(null, new object[] { BoForUnityManager.BoTorchKernelPreset.SharedAddInterMatern32 }),
                    Is.EqualTo("shared_add_inter_matern32")
                );
                Assert.That(
                    normalizeMethod.Invoke(null, new object[] { BoForUnityManager.BoTorchKernelPreset.Matern32DimScaledArd }),
                    Is.EqualTo("matern32_dimscaled_ard")
                );

                // Unknown parent key
                manager.parameters[1].value.gatedBy = "missing";
                Assert.That(string.Join(" ", BoConfigValidator.GetErrors(manager)), Does.Contain("does not match"));

                // Self-gating
                manager.parameters[1].value.gatedBy = "size";
                Assert.That(string.Join(" ", BoConfigValidator.GetErrors(manager)), Does.Contain("itself"));

                // Cycle: alpha -> size -> alpha
                manager.parameters[1].value.gatedBy = "alpha";
                manager.parameters[0].value.gatedBy = "size";
                Assert.That(string.Join(" ", BoConfigValidator.GetErrors(manager)), Does.Contain("cycle"));

                // Out-of-range threshold / neutral
                manager.parameters[0].value.gatedBy = "";
                manager.parameters[1].value.gateThreshold = 1.0f;
                manager.parameters[1].value.gateNeutral = 1.5f;
                string errs = string.Join(" ", BoConfigValidator.GetErrors(manager));
                Assert.That(errs, Does.Contain("Gate Threshold"));
                Assert.That(errs, Does.Contain("Gate Neutral"));
            }
            finally { UnityEngine.Object.DestroyImmediate(target); }
        }
    }
}
