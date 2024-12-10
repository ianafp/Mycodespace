import mitsuba as mi
mi.set_variant('scalar_rgb')
from mitsuba import Float, Vector3f, Spectrum
import drjit as dr
import numpy as np
import imageio
# import taichi as ti 
# ti.init(arch=ti.gpu)

class PathIntegrator(mi.SamplingIntegrator):
    def __init__(self, props):
        super().__init__(props)
        self.m_max_depth = props.get('max_depth', 5)  # Default max depth
        self.m_rr_depth = props.get('rr_depth', 3)    # Default Russian roulette depth
        self.m_hide_emitters = props.get('hide_emitters', False)

    def sample(self, scene, sampler, ray_, medium=None, aovs=None, active=True):
        if self.m_max_depth == 0:
            return Float(0), False

        # --------------------- Configure loop state ----------------------
        ray = mi.Ray3f(ray_)
        throughput = mi.Spectrum(1)
        result = mi.Spectrum(0)
        eta = mi.ScalarFloat(1)
        depth = 0

        valid_ray = not self.m_hide_emitters and scene.environment() is not None

        prev_si = mi.SurfaceInteraction3f()  # Previous surface interaction
        prev_bsdf_pdf = mi.ScalarFloat(1)
        prev_bsdf_delta = True
        bsdf_ctx = mi.BSDFContext()

        # LoopState for JIT-style optimization
        class LoopState:
            def __init__(self):
                self.ray = ray
                self.throughput = throughput
                self.result = result
                self.eta = eta
                self.depth = depth
                self.valid_ray = valid_ray
                self.prev_si = prev_si
                self.prev_bsdf_pdf = prev_bsdf_pdf
                self.prev_bsdf_delta = prev_bsdf_delta
                self.active = active
                self.sampler = sampler

        ls = LoopState()

        while ls.active:
            si = scene.ray_intersect(ls.ray, ray_flags=mi.RayFlags.All, coherent=ls.depth == 0)

            # Direct emission handling
            if si.emitter(scene) is not None:
                ds = mi.DirectionSample3f(scene, si, ls.prev_si)
                em_pdf = mi.ScalarFloat(0)

                if not ls.prev_bsdf_delta:
                    em_pdf = scene.pdf_emitter_direction(ls.prev_si, ds, not ls.prev_bsdf_delta)

                mis_bsdf = self.mis_weight(ls.prev_bsdf_pdf, em_pdf)
                ls.result += ls.throughput * ds.emitter.eval(si) * mis_bsdf

            # Continue tracing the path?
            active_next = (ls.depth + 1 < self.m_max_depth) and si.is_valid()

            if not active_next:
                ls.active = active_next
                break  # Exit early if not active

            bsdf = si.bsdf(ls.ray)

            # Emitter sampling
            bsdf_flag = bsdf.flags()
            is_bsdf_smooth = bsdf_flag & mi.BSDFFlags.Smooth > 0
            active_em = active_next and is_bsdf_smooth
            ds = mi.DirectionSample3f()
            em_weight = mi.Spectrum(0.0)
            wo = mi.Vector3f(0.0,0.0,0.0)

            if active_em:
                ds, em_weight = scene.sample_emitter_direction(si, ls.sampler.next_2d(), True)
                active_em &= (ds.pdf != 0.0)
                wo = si.to_local(ds.d)
                # if ds.pdf != 0:
                #     em_weight = scene.eval_emitter_direction(si, ds)

            # wo = si.to_local(ds.d) if active_em else si.to_local(ds.d)

            # BSDF evaluation and sampling
            sample_1 = ls.sampler.next_1d()
            sample_2 = ls.sampler.next_2d()
            bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2)

            if active_em:
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)
                mis_em = self.mis_weight(ds.pdf, bsdf_pdf)
                ls.result += ls.throughput * bsdf_val * em_weight * mis_em

            # Update ray and throughput
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)
            ls.ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            ls.throughput *= bsdf_weight
            ls.eta *= bsdf_sample.eta
            bsdf_sample_flag = bsdf_flag
            is_bsdf_null = (bsdf_sample_flag & mi.BSDFFlags.Null) > 0
            ls.valid_ray |= ls.active and si.is_valid() and not is_bsdf_null
            # Update previous interaction
            ls.prev_si = si
            ls.prev_bsdf_pdf = bsdf_sample.pdf
            is_bsdf_delta = (bsdf_sample_flag & mi.BSDFFlags.Delta) > 0
            ls.prev_bsdf_delta = is_bsdf_delta

            # Depth increment and Russian roulette
            if si.is_valid():
                ls.depth += 1

            throughput_max = max(ls.throughput)

            rr_prob = min(throughput_max * ls.eta * ls.eta+1e-6, 0.95)
            rr_active = ls.depth >= self.m_rr_depth
            rr_continue = ls.sampler.next_1d() < rr_prob

            ls.throughput *= 1 / rr_prob if rr_active else 1
            ls.active = active_next and (not rr_active or rr_continue) and throughput_max != 0

        return ls.result if ls.valid_ray else mi.Spectrum(0), ls.valid_ray

    def mis_weight(self, pdf_a, pdf_b):
        pdf_a *= pdf_a
        pdf_b *= pdf_b
        return pdf_a / (pdf_a + pdf_b)

    def to_string(self):
        return f"PathIntegrator[max_depth={self.m_max_depth}, rr_depth={self.m_rr_depth}]"


# 注册自定义积分器
mi.register_integrator('ptracer', PathIntegrator)
# 渲染函数

# 加载场景和执行渲染
scene = mi.load_file('C:\\Users\\86173\\Documents\\code\\CG\\mitsuba\\scenes\\cbox.xml')
sensor = scene.sensors()[0]
sampler = mi.load_dict({
    'type': 'independent',
    'sample_count': 128
})
integrator = mi.load_dict({
    'type':'ptracer',
    'max_depth': 8,
    'rr_depth': 5,
    'hide_emitters': False
})

# 图像分辨率
width, height = 512,512

# 初始化图像结果
image_result = np.zeros((height, width, 3), dtype=np.float32)
spp = 16
# 创建渲染循环：遍历每个像素，手动调用 `sample` 函数

# for y,x in range(wid)
# @ti.kernel
def render():
    for y in range(height):
        for x in range(width):
            color_sum = mi.Spectrum(0)
            for k in range(spp):
                # 构造当前像素的射线（假设有摄像机设置）
                # 这里应该有摄像机投影来生成射线
                # 将像素坐标(x, y)转换为归一化的传感器坐标
                sample2d = mi.Point2f((x + 0.5) / width, (y + 0.5) / height)

                # 从传感器生成光线
                time = 0  # 通常时间设置为 0 或 1
                sample1 = sampler.next_1d()  # 第一个采样值
                sample2 = sampler.next_2d()  # 第二个采样值（2D）
                sample3 = sampler.next_2d()  # 第三个采样值（2D）

                # 使用 sample_ray 函数生成光线
                sensor_ray, _ = sensor.sample_ray(time, sample1, sample2d, sample3)
                # 生成光线
                # sensor_ray, _ = sensor.sample_ray(time, sample1, sample2, sample3)
                # 调用 sample 函数获取当前像素的光照结果
                color, valid = integrator.sample(scene, sampler, sensor_ray)
                # print('y,x:{},{},color:{},{},{}'.format(y,x,c olor[0],color[1],color[2]))
                color_sum += color
            # 将采样结果累积到图像
            image_result[y, x] = color_sum/spp

# 保存渲染结果到图像文件
from PIL import Image as PILImage
render()

image = PILImage.fromarray((image_result * 255).astype(np.uint8))
image.save("rendered_image.png")
print("Rendering completed and saved as 'rendered_image.png'")

# image = mi.render(scene,spp = 256)
# image = np.asarray(image)
# image = PILImage.fromarray((image * 255).astype(np.uint8))
# image.save("rendered_image_p.png")
# print("Rendering completed and saved as 'rendered_image_t.png'")