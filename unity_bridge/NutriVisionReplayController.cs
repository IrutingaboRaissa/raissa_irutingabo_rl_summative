using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

[Serializable]
public class ReplayStep
{
    public int step;
    public int action_index;
    public string action_name;
    public float reward;
    public string food_name;
    public float daily_calories;
    public float daily_protein;
    public int num_meals;
}

[Serializable]
public class ReplayEpisode
{
    public int episode_index;
    public string goal_name;
    public float total_reward;
    public int num_steps;
    public List<ReplayStep> steps;
}

[Serializable]
public class ReplayPayload
{
    public string algorithm;
    public string best_config;
    public List<string> action_names;
    public List<ReplayEpisode> episodes;
}

public class NutriVisionReplayController : MonoBehaviour
{
    public enum RunMode
    {
        ReplayFromJson,
        InteractiveUnityEnv
    }

    [Header("Mode")]
    public RunMode runMode = RunMode.ReplayFromJson;

    [Header("Replay Data")]
    public string jsonFileName = "replay_trajectories.json";
    public float secondsPerStep = 0.8f;

    [Header("Scene References")]
    public Transform agentVisual;
    public TextMeshProUGUI summaryText;
    public TextMeshProUGUI stepText;
    public TextMeshProUGUI helpText;
    public Image caloriesBar;
    public Image proteinBar;
    public bool autoCreateDemoScene = true;

    private ReplayPayload _payload;
    private int _episodeIndex = 0;
    private int _stepIndex = 0;
    private NutriVisionUnityEnv _interactiveEnv;
    private bool _paused = false;
    private Coroutine _replayCoroutine;

    private void Start()
    {
        if (autoCreateDemoScene)
        {
            EnsureDemoScene();
        }

        if (runMode == RunMode.ReplayFromJson)
        {
            LoadReplay();
            _replayCoroutine = StartCoroutine(PlayReplay());
            SetHelp("Replay Mode\nSpace: Pause/Resume | N: Next episode | R: Restart");
        }
        else
        {
            _interactiveEnv = new NutriVisionUnityEnv(seed: 12345);
            _interactiveEnv.Reset();
            RenderInteractiveState("Reset");
            SetHelp("Interactive Mode\nPress 1: Accept | 2: Lower Calorie | 3: Higher Protein | 4: Skip\nR: Reset episode");
        }
    }

    private void Update()
    {
        if (runMode == RunMode.ReplayFromJson)
        {
            HandleReplayInput();
        }
        else
        {
            HandleInteractiveInput();
        }
    }

    private void LoadReplay()
    {
        string path = Path.Combine(Application.streamingAssetsPath, jsonFileName);
        if (!File.Exists(path))
        {
            Debug.LogError("Replay JSON not found: " + path);
            return;
        }

        string json = File.ReadAllText(path);
        _payload = JsonUtility.FromJson<ReplayPayload>(json);

        if (_payload == null || _payload.episodes == null || _payload.episodes.Count == 0)
        {
            Debug.LogError("Replay JSON is empty or invalid.");
            return;
        }

        if (summaryText != null)
        {
            summaryText.text = $"Algorithm: {_payload.algorithm.ToUpper()}";
        }
    }

    private IEnumerator PlayReplay()
    {
        if (_payload == null || _payload.episodes == null || _payload.episodes.Count == 0)
            yield break;

        while (_episodeIndex < _payload.episodes.Count)
        {
            ReplayEpisode ep = _payload.episodes[_episodeIndex];

            _stepIndex = 0;
            while (_stepIndex < ep.steps.Count)
            {
                ReplayStep st = ep.steps[_stepIndex];
                RenderStep(ep, st);
                _stepIndex++;
                float waited = 0f;
                while (waited < secondsPerStep)
                {
                    if (!_paused) waited += Time.deltaTime;
                    yield return null;
                }
            }

            _episodeIndex++;
            yield return new WaitForSeconds(1.0f);
        }
    }

    private void RenderStep(ReplayEpisode ep, ReplayStep st)
    {
        if (agentVisual != null)
        {
            // Simple visual movement to show progress across steps.
            float x = st.step * 0.25f;
            float y = Mathf.Clamp(st.reward, -2f, 6f) * 0.15f;
            agentVisual.position = new Vector3(x, y, 0f);
        }

        if (stepText != null)
        {
            stepText.text =
                $"Episode {ep.episode_index + 1} ({ep.goal_name})\n" +
                $"Step: {st.step + 1}/{ep.num_steps}\n" +
                $"Action: {st.action_name}\n" +
                $"Food: {st.food_name}\n" +
                $"Reward: {st.reward:F2}\n" +
                $"Calories: {st.daily_calories:F0}\n" +
                $"Protein: {st.daily_protein:F1}g\n" +
                $"Meals Logged: {st.num_meals}";
        }
        UpdateBars(st.daily_calories, st.daily_protein, 2500f, 160f);
    }

    private void HandleReplayInput()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            _paused = !_paused;
            SetHelp(_paused
                ? "Replay Paused\nSpace: Resume | N: Next episode | R: Restart"
                : "Replay Mode\nSpace: Pause | N: Next episode | R: Restart");
        }
        if (Input.GetKeyDown(KeyCode.R))
        {
            if (_replayCoroutine != null) StopCoroutine(_replayCoroutine);
            _episodeIndex = 0;
            _stepIndex = 0;
            _paused = false;
            _replayCoroutine = StartCoroutine(PlayReplay());
        }
        if (Input.GetKeyDown(KeyCode.N))
        {
            _episodeIndex = Mathf.Min(_episodeIndex + 1, (_payload?.episodes?.Count ?? 1) - 1);
            _stepIndex = 0;
        }
    }

    private void HandleInteractiveInput()
    {
        int action = -1;
        if (Input.GetKeyDown(KeyCode.Alpha1)) action = 0;
        if (Input.GetKeyDown(KeyCode.Alpha2)) action = 1;
        if (Input.GetKeyDown(KeyCode.Alpha3)) action = 2;
        if (Input.GetKeyDown(KeyCode.Alpha4)) action = 3;

        if (Input.GetKeyDown(KeyCode.R))
        {
            _interactiveEnv.Reset();
            RenderInteractiveState("Reset");
            return;
        }

        if (action == -1) return;

        var (reward, terminated, actionName) = _interactiveEnv.Step(action);
        RenderInteractiveState(actionName, reward);

        if (terminated)
        {
            SetHelp($"Episode finished (total reward: {_interactiveEnv.episodeReward:F2}). Press R to reset.");
        }
    }

    private void RenderInteractiveState(string actionName, float reward = 0f)
    {
        if (summaryText != null)
        {
            summaryText.text = $"Interactive NutriVision\nGoal: {_interactiveEnv.goalName}\nTotal Reward: {_interactiveEnv.episodeReward:F2}";
        }

        if (stepText != null)
        {
            stepText.text =
                $"Step: {_interactiveEnv.stepCount}/{_interactiveEnv.maxSteps}\n" +
                $"Action: {actionName}\n" +
                $"Current Food: {_interactiveEnv.currentFood.name}\n" +
                $"Reward: {reward:F2}\n" +
                $"Calories: {_interactiveEnv.dailyCalories:F0}/{_interactiveEnv.calorieTarget:F0}\n" +
                $"Protein: {_interactiveEnv.dailyProtein:F1}/{_interactiveEnv.proteinTarget:F1}\n" +
                $"Meals Logged: {_interactiveEnv.numMealsLogged}";
        }

        if (agentVisual != null)
        {
            float x = _interactiveEnv.stepCount * 0.25f;
            float y = Mathf.Clamp(_interactiveEnv.episodeReward, -5f, 40f) * 0.03f;
            agentVisual.position = new Vector3(x, y, 0f);
        }

        UpdateBars(
            _interactiveEnv.dailyCalories,
            _interactiveEnv.dailyProtein,
            _interactiveEnv.calorieTarget,
            _interactiveEnv.proteinTarget
        );
    }

    private void UpdateBars(float calories, float protein, float calorieTarget, float proteinTarget)
    {
        if (caloriesBar != null)
        {
            float c = calorieTarget > 0f ? calories / calorieTarget : 0f;
            caloriesBar.fillAmount = Mathf.Clamp01(c);
            caloriesBar.color = c > 1f ? new Color(0.93f, 0.23f, 0.23f) : new Color(0.13f, 0.77f, 0.37f);
        }
        if (proteinBar != null)
        {
            float p = proteinTarget > 0f ? protein / proteinTarget : 0f;
            proteinBar.fillAmount = Mathf.Clamp01(p);
            proteinBar.color = p > 1f ? new Color(0.93f, 0.23f, 0.23f) : new Color(0.23f, 0.51f, 0.96f);
        }
    }

    private void SetHelp(string text)
    {
        if (helpText != null) helpText.text = text;
    }

    private void EnsureDemoScene()
    {
        EnsureGround();
        EnsureAgentVisual();
        EnsureUi();
    }

    private void EnsureGround()
    {
        GameObject ground = GameObject.Find("Ground");
        if (ground != null) return;

        ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.position = new Vector3(2f, -0.51f, 0f);
        ground.transform.localScale = new Vector3(1.2f, 1f, 0.5f);
    }

    private void EnsureAgentVisual()
    {
        if (agentVisual != null) return;

        GameObject avatar = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        avatar.name = "AgentAvatar";
        avatar.transform.position = new Vector3(0f, 0f, 0f);

        Renderer r = avatar.GetComponent<Renderer>();
        if (r != null)
        {
            r.material.color = new Color(0.25f, 0.6f, 0.95f);
        }

        agentVisual = avatar.transform;
    }

    private void EnsureUi()
    {
        Canvas canvas = FindObjectOfType<Canvas>();
        if (canvas == null)
        {
            GameObject canvasGo = new GameObject("Canvas", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvas = canvasGo.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            CanvasScaler scaler = canvasGo.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
        }

        if (summaryText == null)
        {
            summaryText = CreateTmpText(canvas.transform, "SummaryText", new Vector2(20, -20), new Vector2(620, 130), 28);
        }
        if (stepText == null)
        {
            stepText = CreateTmpText(canvas.transform, "StepText", new Vector2(20, -170), new Vector2(620, 420), 24);
        }
        if (helpText == null)
        {
            helpText = CreateTmpText(canvas.transform, "HelpText", new Vector2(20, -620), new Vector2(900, 140), 22);
        }

        if (caloriesBar == null)
        {
            caloriesBar = CreateBar(canvas.transform, "CaloriesBar", new Vector2(700, -70), new Vector2(480, 28), new Color(0.13f, 0.77f, 0.37f));
            CreateTmpText(canvas.transform, "CaloriesLabel", new Vector2(700, -40), new Vector2(500, 24), 20).text = "Calories Progress";
        }

        if (proteinBar == null)
        {
            proteinBar = CreateBar(canvas.transform, "ProteinBar", new Vector2(700, -150), new Vector2(480, 28), new Color(0.23f, 0.51f, 0.96f));
            CreateTmpText(canvas.transform, "ProteinLabel", new Vector2(700, -120), new Vector2(500, 24), 20).text = "Protein Progress";
        }
    }

    private TextMeshProUGUI CreateTmpText(Transform parent, string objName, Vector2 anchoredPos, Vector2 size, int fontSize)
    {
        GameObject go = new GameObject(objName, typeof(RectTransform), typeof(TextMeshProUGUI));
        go.transform.SetParent(parent, false);

        RectTransform rt = go.GetComponent<RectTransform>();
        rt.anchorMin = new Vector2(0f, 1f);
        rt.anchorMax = new Vector2(0f, 1f);
        rt.pivot = new Vector2(0f, 1f);
        rt.anchoredPosition = anchoredPos;
        rt.sizeDelta = size;

        TextMeshProUGUI tmp = go.GetComponent<TextMeshProUGUI>();
        tmp.fontSize = fontSize;
        tmp.color = Color.white;
        tmp.text = objName;
        tmp.alignment = TextAlignmentOptions.TopLeft;
        return tmp;
    }

    private Image CreateBar(Transform parent, string objName, Vector2 anchoredPos, Vector2 size, Color fillColor)
    {
        GameObject bg = new GameObject(objName + "_BG", typeof(RectTransform), typeof(Image));
        bg.transform.SetParent(parent, false);
        RectTransform bgRt = bg.GetComponent<RectTransform>();
        bgRt.anchorMin = new Vector2(0f, 1f);
        bgRt.anchorMax = new Vector2(0f, 1f);
        bgRt.pivot = new Vector2(0f, 1f);
        bgRt.anchoredPosition = anchoredPos;
        bgRt.sizeDelta = size;
        Image bgImg = bg.GetComponent<Image>();
        bgImg.color = new Color(0.2f, 0.2f, 0.2f, 0.75f);

        GameObject fill = new GameObject(objName, typeof(RectTransform), typeof(Image));
        fill.transform.SetParent(bg.transform, false);
        RectTransform fillRt = fill.GetComponent<RectTransform>();
        fillRt.anchorMin = new Vector2(0f, 0f);
        fillRt.anchorMax = new Vector2(1f, 1f);
        fillRt.offsetMin = Vector2.zero;
        fillRt.offsetMax = Vector2.zero;

        Image fillImg = fill.GetComponent<Image>();
        fillImg.type = Image.Type.Filled;
        fillImg.fillMethod = Image.FillMethod.Horizontal;
        fillImg.fillOrigin = (int)Image.OriginHorizontal.Left;
        fillImg.fillAmount = 0f;
        fillImg.color = fillColor;
        return fillImg;
    }
}

