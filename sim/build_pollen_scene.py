import io
src = io.open("pollen_robot.xml", encoding="utf-8").read()
# Their model is the robot alone: no floor, no sim options. Add exactly those.
add = '''  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" />
  <worldbody>
    <geom name="floor" type="plane" size="6 6 0.05" pos="0 0 0" rgba="0.16 0.18 0.16 1"
          contype="1" conaffinity="1" condim="3" friction="1.0 0.02 0.001" />
  </worldbody>
'''
src = src.replace("<worldbody>", add + "  <worldbody>", 1)
io.open("scene_pollen.xml", "w", encoding="utf-8").write(src)
print("scene_pollen.xml written")
