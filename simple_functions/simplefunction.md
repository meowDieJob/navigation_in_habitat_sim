# 碰撞检测
```python
    nav_point = agent_state.position
    max_search_radius = 0.25
    hit_record = env.pathfinder.closest_obstacle_surface_point(
        nav_point, max_search_radius
    )
	
    print("Closest obstacle HitRecord:")
    print(" point: " + str(hit_record.hit_pos))
    print(" normal: " + str(hit_record.hit_normal))
    print(" distance: " + str(hit_record.hit_dist))
```
其中的```python 
hit_record.hitdist```等于0.0时即发生碰撞
#
# 寻找某个到导航点的最优动作（路径）
```python
follower = env.make_greedy_follower(agent_id=0, goal_radius=0.25,stop_key="Stop")


```
然后使用以下代码获取到nav_point的最优动作（下一步）,但是会有概率报这个错，跟地图有关
可以在官方文档查看更多函数
```python
try:
    action = follower.next_action_along(goal_pos = nav_point)
except habitat_sim.errors.GreedyFollowerError:
    break
```
