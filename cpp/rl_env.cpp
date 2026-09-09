#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <array>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <sstream>

namespace py = pybind11;

struct Vec2
{
    float x, y;
};

class Player
{
public:
    Vec2 position{0.0f, 0.0f};
    Vec2 velocity{0.0f, 0.0f};
    int team = 0;
    float maxSpeed = 6.0f;
    float radius = 0.5f;
    bool hasBall = false;
};

class Ball
{
public:
    Vec2 position{0.0f, 0.0f};
    Vec2 velocity{0.0f, 0.0f};
    float radius = 0.35f;
    float drag = 0.985f;
    float friction = 0.98f;
};

class Field
{
public:
    float width = 60.0f;
    float height = 40.0f;
    float goalWidth = 7.0f;

    bool inside(const Vec2& p, float margin = 0.0f) const
    {
        return p.x >= -width * 0.5f + margin && p.x <= width * 0.5f - margin &&
               p.y >= -height * 0.5f + margin && p.y <= height * 0.5f - margin;
    }
};

class TeamSportsEnv
{
public:
    Field field;
    Ball ball;
    std::vector<Player> players;
    int aiPlayerIndex = 0;
    int teammateIndex = 1;
    int opponentIndex = 2;

    int teamAScore = 0;
    int teamBScore = 0;
    float timeRemaining = 60.0f;
    float stepDuration = 1.0f / 15.0f;
    bool terminated = false;
    float lastReward = 0.0f;
    int winner = 0; // 0 = none, 1 = team A, 2 = team B

    std::mt19937 rng;

    TeamSportsEnv()
    {
        rng.seed(0);
        reset();
    }

    void reset()
    {
        players.clear();

        // Team A: AI field player
        Player ai;
        ai.team = 0;
        ai.position = {-10.0f, 0.0f};
        players.push_back(ai);

        // Team A: teammate
        Player teammate;
        teammate.team = 0;
        teammate.position = {-5.0f, 5.0f};
        players.push_back(teammate);

        // Team B: opponent
        Player opponent;
        opponent.team = 1;
        opponent.position = {10.0f, 0.0f};
        players.push_back(opponent);

        ball.position = {0.0f, 0.0f};
        ball.velocity = {0.0f, 0.0f};

        teamAScore = 0;
        teamBScore = 0;
        timeRemaining = 60.0f;
        terminated = false;
        winner = 0;
        lastReward = 0.0f;

        assignPossession();
    }

    std::vector<float> getObservation()
    {
        const Player& self = players[aiPlayerIndex];
        const Player& teammate = players[teammateIndex];
        const Player& opponent = players[opponentIndex];

        Vec2 ownGoal = ownGoalPosition(self.team);
        Vec2 enemyGoal = enemyGoalPosition(self.team);

        std::vector<float> obs;
        obs.reserve(20);

        // Self state (normalized to roughly [-1, 1])
        obs.push_back(self.position.x / (field.width * 0.5f));
        obs.push_back(self.position.y / (field.height * 0.5f));
        obs.push_back(self.velocity.x / self.maxSpeed);
        obs.push_back(self.velocity.y / self.maxSpeed);

        // Ball relative
        obs.push_back((ball.position.x - self.position.x) / field.width);
        obs.push_back((ball.position.y - self.position.y) / field.height);
        obs.push_back(ball.velocity.x / 20.0f);
        obs.push_back(ball.velocity.y / 20.0f);

        // Teammate relative
        obs.push_back((teammate.position.x - self.position.x) / field.width);
        obs.push_back((teammate.position.y - self.position.y) / field.height);
        obs.push_back(teammate.velocity.x / teammate.maxSpeed);
        obs.push_back(teammate.velocity.y / teammate.maxSpeed);

        // Opponent relative
        obs.push_back((opponent.position.x - self.position.x) / field.width);
        obs.push_back((opponent.position.y - self.position.y) / field.height);
        obs.push_back(opponent.velocity.x / opponent.maxSpeed);
        obs.push_back(opponent.velocity.y / opponent.maxSpeed);

        // Goals relative
        obs.push_back((ownGoal.x - self.position.x) / field.width);
        obs.push_back((ownGoal.y - self.position.y) / field.height);
        obs.push_back((enemyGoal.x - self.position.x) / field.width);
        obs.push_back((enemyGoal.y - self.position.y) / field.height);

        return obs;
    }

    py::array_t<float> getObservationNp()
    {
        auto obs = getObservation();
        return vectorToNp(obs);
    }

    std::tuple<py::array_t<float>, float, bool> step(py::array_t<float> action)
    {
        auto buf = action.request();
        if (buf.size < 3)
            throw std::runtime_error("Action array must have at least 3 elements: [move_x, move_y, kick]");

        const float* data = static_cast<const float*>(buf.ptr);
        float moveX = std::clamp(data[0], -1.0f, 1.0f);
        float moveY = std::clamp(data[1], -1.0f, 1.0f);
        bool kick = data[2] > 0.5f;

        advance(moveX, moveY, kick);

        auto obs = getObservationNp();
        return std::make_tuple(obs, lastReward, terminated);
    }

    std::tuple<py::array_t<float>, float, bool> resetAndGetObs()
    {
        reset();
        return std::make_tuple(getObservationNp(), 0.0f, false);
    }

    std::string getDebugText()
    {
        std::ostringstream oss;
        const Player& self = players[aiPlayerIndex];
        oss << "AI:      (" << self.position.x << ", " << self.position.y << ")\n";
        oss << "Ball:    (" << ball.position.x << ", " << ball.position.y << ")\n";
        oss << "BallVel: (" << ball.velocity.x << ", " << ball.velocity.y << ")\n";
        oss << "Teammate:(" << players[teammateIndex].position.x << ", " << players[teammateIndex].position.y << ")\n";
        oss << "Opponent:(" << players[opponentIndex].position.x << ", " << players[opponentIndex].position.y << ")\n";
        oss << "Score:   A=" << teamAScore << " B=" << teamBScore << " | Time: " << timeRemaining << "\n";
        return oss.str();
    }

private:
    void advance(float moveX, float moveY, bool kick)
    {
        if (terminated)
        {
            lastReward = 0.0f;
            return;
        }

        // Apply AI action to self
        Player& self = players[aiPlayerIndex];
        self.velocity.x += moveX * 40.0f * stepDuration;
        self.velocity.y += moveY * 40.0f * stepDuration;

        // Scripted teammate: support position around ball-carrier or ball
        Player& teammate = players[teammateIndex];
        Vec2 target = supportTarget(teammate.team);
        moveToward(teammate, target);

        // Scripted opponent: move toward ball
        Player& opponent = players[opponentIndex];
        moveToward(opponent, ball.position);

        integratePlayers();
        integrateBall();
        handleCollisions();
        assignPossession();

        if (kick && self.hasBall)
            kickBall(self, enemyGoalPosition(self.team));

        int goalTeam = checkGoal();
        float reward = 0.0f;
        if (goalTeam == self.team)
        {
            reward += 100.0f;
            teamAScore += (self.team == 0) ? 1 : 0;
            teamBScore += (self.team == 1) ? 1 : 0;
            terminated = true;
            winner = self.team + 1;
        }
        else if (goalTeam >= 0)
        {
            reward -= 100.0f;
            teamAScore += (self.team == 1) ? 1 : 0;
            teamBScore += (self.team == 0) ? 1 : 0;
            terminated = true;
            winner = (self.team == 0 ? 2 : 1);
        }

        if (!terminated)
        {
            timeRemaining -= stepDuration;
            if (timeRemaining <= 0.0f)
            {
                terminated = true;
                if (teamAScore > teamBScore)
                    winner = 1;
                else if (teamBScore > teamAScore)
                    winner = 2;
            }
        }

        reward += computeSupportReward();
        lastReward = reward;
    }

    Vec2 supportTarget(int team)
    {
        // Stay between ball and enemy goal, slightly behind ball, on a useful passing lane
        Vec2 goal = enemyGoalPosition(team);
        Vec2 own = ownGoalPosition(team);
        Vec2 target;
        target.x = ball.position.x + (ball.velocity.x * 0.3f);
        target.y = ball.position.y + (ball.velocity.y * 0.3f);

        // Pull toward lateral support offset
        float offsetY = (team == 0 ? 4.0f : -4.0f);
        target.y += offsetY;

        // Don't run too far ahead of the ball
        float ballToGoalX = goal.x - ball.position.x;
        float progress = (target.x - own.x) / (goal.x - own.x + 1e-6f);
        float ballProgress = (ball.position.x - own.x) / (goal.x - own.x + 1e-6f);
        if (progress > ballProgress + 0.25f)
            target.x = ball.position.x + 0.25f * ballToGoalX;

        return target;
    }

    void moveToward(Player& p, const Vec2& target)
    {
        Vec2 dir{target.x - p.position.x, target.y - p.position.y};
        float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        if (len > 1e-6f)
        {
            dir.x /= len;
            dir.y /= len;
        }
        p.velocity.x += dir.x * 30.0f * stepDuration;
        p.velocity.y += dir.y * 30.0f * stepDuration;
    }

    void integratePlayers()
    {
        for (auto& p : players)
        {
            float speed = std::sqrt(p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y);
            if (speed > p.maxSpeed)
            {
                p.velocity.x *= p.maxSpeed / speed;
                p.velocity.y *= p.maxSpeed / speed;
            }

            p.position.x += p.velocity.x * stepDuration;
            p.position.y += p.velocity.y * stepDuration;

            // Simple drag
            p.velocity.x *= 0.9f;
            p.velocity.y *= 0.9f;

            // Field bounds
            if (!field.inside(p.position, p.radius))
            {
                p.position.x = std::clamp(p.position.x, -field.width * 0.5f + p.radius, field.width * 0.5f - p.radius);
                p.position.y = std::clamp(p.position.y, -field.height * 0.5f + p.radius, field.height * 0.5f - p.radius);
                p.velocity = {0.0f, 0.0f};
            }
        }
    }

    void integrateBall()
    {
        ball.position.x += ball.velocity.x * stepDuration;
        ball.position.y += ball.velocity.y * stepDuration;

        ball.velocity.x *= ball.drag;
        ball.velocity.y *= ball.drag;

        // Field bounds (bounce)
        if (ball.position.x < -field.width * 0.5f + ball.radius || ball.position.x > field.width * 0.5f - ball.radius)
        {
            ball.velocity.x *= -0.6f;
            ball.position.x = std::clamp(ball.position.x, -field.width * 0.5f + ball.radius, field.width * 0.5f - ball.radius);
        }

        bool inGoalY = ball.position.y > -field.goalWidth * 0.5f && ball.position.y < field.goalWidth * 0.5f;
        if (!inGoalY)
        {
            if (ball.position.y < -field.height * 0.5f + ball.radius || ball.position.y > field.height * 0.5f - ball.radius)
            {
                ball.velocity.y *= -0.6f;
                ball.position.y = std::clamp(ball.position.y, -field.height * 0.5f + ball.radius, field.height * 0.5f - ball.radius);
            }
        }
    }

    void handleCollisions()
    {
        for (auto& p : players)
        {
            float dx = ball.position.x - p.position.x;
            float dy = ball.position.y - p.position.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            float minDist = p.radius + ball.radius;
            if (dist < minDist && dist > 1e-6f)
            {
                float overlap = minDist - dist;
                float nx = dx / dist;
                float ny = dy / dist;
                ball.position.x += nx * overlap;
                ball.position.y += ny * overlap;

                // Add player's velocity to ball, with a small bump
                ball.velocity.x += p.velocity.x * 0.3f + nx * 2.0f;
                ball.velocity.y += p.velocity.y * 0.3f + ny * 2.0f;
            }
        }
    }

    void assignPossession()
    {
        for (auto& p : players)
            p.hasBall = false;

        float bestDist = 1e9f;
        int owner = -1;
        for (int i = 0; i < static_cast<int>(players.size()); ++i)
        {
            const auto& p = players[i];
            float dx = ball.position.x - p.position.x;
            float dy = ball.position.y - p.position.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < (p.radius + ball.radius + 0.15f) && dist < bestDist)
            {
                bestDist = dist;
                owner = i;
            }
        }

        if (owner >= 0)
            players[owner].hasBall = true;
    }

    void kickBall(const Player& p, const Vec2& target)
    {
        Vec2 dir{target.x - ball.position.x, target.y - ball.position.y};
        float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        if (len > 1e-6f)
        {
            dir.x /= len;
            dir.y /= len;
        }
        float power = 18.0f;
        ball.velocity.x = dir.x * power + p.velocity.x * 0.5f;
        ball.velocity.y = dir.y * power + p.velocity.y * 0.5f;
        players[aiPlayerIndex].hasBall = false;
    }

    // Returns team index that scored, or -1 if no goal
    int checkGoal()
    {
        bool inGoalY = ball.position.y > -field.goalWidth * 0.5f && ball.position.y < field.goalWidth * 0.5f;
        if (!inGoalY)
            return -1;

        if (ball.position.x < -field.width * 0.5f)
            return 1; // Team B scores
        if (ball.position.x > field.width * 0.5f)
            return 0; // Team A scores

        return -1;
    }

    Vec2 ownGoalPosition(int team) const
    {
        return {team == 0 ? -field.width * 0.5f : field.width * 0.5f, 0.0f};
    }

    Vec2 enemyGoalPosition(int team) const
    {
        return {team == 0 ? field.width * 0.5f : -field.width * 0.5f, 0.0f};
    }

    float computeSupportReward()
    {
        const Player& self = players[aiPlayerIndex];
        const Player& teammate = players[teammateIndex];

        float reward = 0.0f;

        // Small penalty for doing nothing to encourage movement
        float speed = std::sqrt(self.velocity.x * self.velocity.x + self.velocity.y * self.velocity.y);
        if (speed > 0.1f)
            reward += 0.005f;

        // Reward being in a useful support position relative to teammate-with-ball or ball
        if (teammate.hasBall)
        {
            Vec2 support = supportTarget(self.team);
            float dx = self.position.x - support.x;
            float dy = self.position.y - support.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            // Reward closeness to support target, scaled so it peaks near 1.5 units
            reward += 0.5f * std::exp(-dist / 4.0f);
        }
        else
        {
            // When no teammate has ball, reward being between ball and enemy goal
            Vec2 goal = enemyGoalPosition(self.team);
            float progress = (self.position.x - ownGoalPosition(self.team).x) / (goal.x - ownGoalPosition(self.team).x);
            float ballProgress = (ball.position.x - ownGoalPosition(self.team).x) / (goal.x - ownGoalPosition(self.team).x);
            float diff = progress - ballProgress;
            // Prefer slightly behind the ball (support from deep)
            reward += 0.1f * std::exp(-diff * diff / 0.5f);
        }

        // Avoid crowding teammate
        float tdx = self.position.x - teammate.position.x;
        float tdy = self.position.y - teammate.position.y;
        float tdist = std::sqrt(tdx * tdx + tdy * tdy);
        if (tdist < 2.0f)
            reward -= 0.05f;

        return reward;
    }

    py::array_t<float> vectorToNp(const std::vector<float>& v)
    {
        py::array_t<float> result(v.size());
        auto r = result.request();
        float* ptr = static_cast<float*>(r.ptr);
        for (size_t i = 0; i < v.size(); ++i)
            ptr[i] = v[i];
        return result;
    }
};

PYBIND11_MODULE(teamsports_rl, m)
{
    m.doc() = "Minimal team sports RL environment";
    py::class_<TeamSportsEnv>(m, "TeamSportsEnv")
        .def(py::init<>())
        .def("reset", &TeamSportsEnv::resetAndGetObs, "Reset environment and return initial observation")
        .def("step", &TeamSportsEnv::step, "Execute one action and return (observation, reward, done)")
        .def("get_observation", &TeamSportsEnv::getObservationNp, "Get current observation vector")
        .def("debug_text", &TeamSportsEnv::getDebugText, "Get human-readable state for debugging")
        .def_readonly("time_remaining", &TeamSportsEnv::timeRemaining)
        .def_readonly("terminated", &TeamSportsEnv::terminated)
        .def_readonly("team_a_score", &TeamSportsEnv::teamAScore)
        .def_readonly("team_b_score", &TeamSportsEnv::teamBScore);
}
