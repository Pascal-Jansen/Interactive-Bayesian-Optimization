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
        public void ValidatorRefusesEmptyAndBackendMismatchedConfigurations()
        {
            var target = new GameObject("ValidatorCompletenessTest");
            try
            {
                var manager = target.AddComponent<BoForUnityManager>();
                string emptyErrors = string.Join(" | ", BoConfigValidator.GetErrors(manager));
                Assert.That(emptyErrors, Does.Contain("At least one parameter"));
                Assert.That(emptyErrors, Does.Contain("At least one objective"));

                manager.parameters = new List<ParameterEntry>
                {
                    new ParameterEntry("p0", new ParameterArgs(0f, 1f))
                };
                manager.objectives = new List<ObjectiveEntry>
                {
                    new ObjectiveEntry("o0", new ObjectiveArgs(0f, 10f, false, 1))
                };
                Assert.That(BoConfigValidator.GetErrors(manager), Is.Empty);

                manager.SetIterationBudget(0, 5);
                Assert.That(string.Join(" | ", BoConfigValidator.GetErrors(manager)), Does.Contain("Sampling Iterations"));
                manager.SetIterationBudget(4, 5);

                manager.warmStart = true;
                manager.initialParametersDataPath = "";
                manager.initialObjectivesDataPath = "";
                Assert.That(string.Join(" | ", BoConfigValidator.GetErrors(manager)), Does.Contain("Warm Start is enabled"));
                manager.warmStart = false;

                manager.optimizerBackend = BoForUnityManager.OptimizerBackend.MetaTAF;
                Assert.That(string.Join(" | ", BoConfigValidator.GetErrors(manager)), Does.Contain("at least 2 objectives"));

                manager.objectives.Add(new ObjectiveEntry("o1", new ObjectiveArgs(0f, 10f, false, 1)));
                Assert.That(BoConfigValidator.GetErrors(manager), Is.Empty);

                manager.optimizerBackend = BoForUnityManager.OptimizerBackend.DBO;
                Assert.That(string.Join(" | ", BoConfigValidator.GetErrors(manager)), Does.Contain("exactly 1 objective"));

                manager.optimizerBackend = BoForUnityManager.OptimizerBackend.CABOP;
                manager.cabopObjectiveMode = BoForUnityManager.CabopObjectiveMode.SingleObjective;
                Assert.That(string.Join(" | ", BoConfigValidator.GetErrors(manager)), Does.Contain("CABOP SingleObjective"));

                manager.contextualOptimization = true;
                Assert.That(string.Join(" | ", BoConfigValidator.GetErrors(manager)), Does.Contain("Contextual optimization"));
                manager.contextualOptimization = false;

                manager.optimizerBackend = BoForUnityManager.OptimizerBackend.BoTorch;
                manager.mcSamples = 0;
                Assert.That(string.Join(" | ", BoConfigValidator.GetErrors(manager)), Does.Contain("MC Samples"));
            }
            finally { UnityEngine.Object.DestroyImmediate(target); }
        }

        [Test]
        public void NonReloadingAdvanceHidesWelcomePanel()
        {
            var target = new GameObject("WelcomePanelAdvanceTest");
            var welcome = new GameObject("WelcomePanel");
            try
            {
                var manager = target.AddComponent<BoForUnityManager>();
                manager.welcomePanel = welcome;
                manager.reloadSceneOnIterationAdvance = false;
                manager.iterationAdvanceMode = BoForUnityManager.IterationAdvanceMode.NextButton;
                manager.totalIterations = 4;
                manager.currentIteration = 1;

                Assert.That(welcome.activeSelf, Is.True);
                manager.InitializationDone();
                Assert.That(welcome.activeSelf, Is.True, "Welcome panel stays visible until the participant starts round 1.");

                manager.RequestNextIteration();
                Assert.That(welcome.activeSelf, Is.False, "Welcome panel must hide when advancing without reloading the scene.");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(welcome);
                UnityEngine.Object.DestroyImmediate(target);
            }
        }
    }
}
