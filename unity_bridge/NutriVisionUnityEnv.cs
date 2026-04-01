using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class FoodItem
{
    public string name;
    public float cal;
    public float protein;
    public float carbs;
    public float fat;
}

public class NutriVisionUnityEnv
{
    public int maxSteps = 20;

    public int goalType; // 0 loss, 1 gain, 2 maintenance
    public string goalName;

    public float calorieTarget;
    public float proteinTarget;
    public float carbsTarget;
    public float fatsTarget;

    public float dailyCalories;
    public float dailyProtein;
    public float dailyCarbs;
    public float dailyFats;
    public int numMealsLogged;
    public int stepCount;
    public float episodeReward;

    public FoodItem currentFood;

    private readonly List<FoodItem> _foods = new List<FoodItem>();
    private readonly System.Random _rng;

    public NutriVisionUnityEnv(int seed = 12345)
    {
        _rng = new System.Random(seed);
        BuildFoodDatabase();
    }

    public void Reset()
    {
        goalType = _rng.Next(0, 3);
        goalName = goalType == 0 ? "Weight Loss" : (goalType == 1 ? "Weight Gain" : "Maintenance");

        if (goalType == 0)
        {
            calorieTarget = 2000f;
            proteinTarget = 150f;
            carbsTarget = 150f;
            fatsTarget = 65f;
        }
        else if (goalType == 1)
        {
            calorieTarget = 3000f;
            proteinTarget = 180f;
            carbsTarget = 300f;
            fatsTarget = 100f;
        }
        else
        {
            calorieTarget = 2500f;
            proteinTarget = 160f;
            carbsTarget = 220f;
            fatsTarget = 80f;
        }

        dailyCalories = 0f;
        dailyProtein = 0f;
        dailyCarbs = 0f;
        dailyFats = 0f;
        numMealsLogged = 0;
        stepCount = 0;
        episodeReward = 0f;
        SampleFood();
    }

    public (float reward, bool terminated, string actionName) Step(int action)
    {
        string actionName = ActionName(action);
        float reward = ComputeReward(action);
        episodeReward += reward;

        if (action == 0)
        {
            dailyCalories += currentFood.cal;
            dailyProtein += currentFood.protein;
            dailyCarbs += currentFood.carbs;
            dailyFats += currentFood.fat;
            numMealsLogged += 1;
        }

        stepCount += 1;
        bool terminated = stepCount >= maxSteps;
        if (dailyCalories > calorieTarget * 1.5f && (goalType == 0 || goalType == 2))
        {
            terminated = true;
        }

        SampleFood();
        return (reward, terminated, actionName);
    }

    private float ComputeReward(int action)
    {
        if (action == 0)
        {
            float calRatio = (dailyCalories + currentFood.cal) / calorieTarget;
            float proteinRatio = (dailyProtein + currentFood.protein) / proteinTarget;
            float carbsRatio = (dailyCarbs + currentFood.carbs) / carbsTarget;
            float fatsRatio = (dailyFats + currentFood.fat) / fatsTarget;

            float calDiff = Mathf.Abs(calRatio - 1.0f);
            float proteinDiff = Mathf.Abs(proteinRatio - 1.0f);
            float carbsDiff = Mathf.Abs(carbsRatio - 1.0f);
            float fatsDiff = Mathf.Abs(fatsRatio - 1.0f);

            float avgDeviation = (calDiff + proteinDiff + carbsDiff + fatsDiff) / 4.0f;
            float reward = 5.0f * (1.0f - Mathf.Min(avgDeviation, 1.0f));

            if (goalType == 0)
            {
                if (calRatio < 1.0f && proteinRatio > 0.8f) reward += 2.0f;
            }
            else if (goalType == 1)
            {
                if (calRatio > 0.8f && proteinRatio > 0.8f) reward += 2.0f;
            }

            return reward;
        }

        if (action == 1 || action == 2) return -0.5f;
        return 0.0f;
    }

    private void SampleFood()
    {
        int idx = _rng.Next(0, _foods.Count);
        currentFood = _foods[idx];
    }

    private static string ActionName(int action)
    {
        if (action == 0) return "Accept";
        if (action == 1) return "Lower Calorie";
        if (action == 2) return "Higher Protein";
        return "Skip";
    }

    private void BuildFoodDatabase()
    {
        _foods.Add(new FoodItem { name = "Jollof Rice", cal = 280, protein = 6, carbs = 48, fat = 5 });
        _foods.Add(new FoodItem { name = "Ndole", cal = 320, protein = 18, carbs = 25, fat = 12 });
        _foods.Add(new FoodItem { name = "Eru", cal = 150, protein = 8, carbs = 12, fat = 7 });
        _foods.Add(new FoodItem { name = "Waakye", cal = 220, protein = 10, carbs = 38, fat = 3 });
        _foods.Add(new FoodItem { name = "Ekwang", cal = 350, protein = 12, carbs = 35, fat = 15 });
        _foods.Add(new FoodItem { name = "Palm-nut Soup", cal = 280, protein = 14, carbs = 15, fat = 18 });
        _foods.Add(new FoodItem { name = "Suya", cal = 320, protein = 35, carbs = 5, fat = 18 });
        _foods.Add(new FoodItem { name = "Injera with Doro", cal = 310, protein = 20, carbs = 40, fat = 8 });
        _foods.Add(new FoodItem { name = "Ugali with Sukuma", cal = 240, protein = 8, carbs = 42, fat = 4 });
        _foods.Add(new FoodItem { name = "Chapati", cal = 280, protein = 7, carbs = 38, fat = 12 });
        _foods.Add(new FoodItem { name = "Fufu", cal = 200, protein = 4, carbs = 44, fat = 1 });
        _foods.Add(new FoodItem { name = "Egusi Soup", cal = 290, protein = 16, carbs = 18, fat = 16 });
        _foods.Add(new FoodItem { name = "Cassava Leaves", cal = 120, protein = 6, carbs = 18, fat = 3 });
        _foods.Add(new FoodItem { name = "Pap with Beans", cal = 260, protein = 12, carbs = 42, fat = 5 });
        _foods.Add(new FoodItem { name = "Yam Porridge", cal = 270, protein = 10, carbs = 45, fat = 6 });
    }
}

