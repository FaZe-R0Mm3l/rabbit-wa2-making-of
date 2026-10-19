# rabbit × 白色相簿2 制作记录

用 Python 合成视频的个人学习项目。这里公开渲染脚本、AI 生成插画、分镜卡点表、逐帧时间轴和历史校验记录。画面围绕《白色相簿2》角色重新制作，并参照 john 的 rabbit MV 学习镜头与节奏。

## 内容

- `outputs/`：21 秒试作脚本，以及 v1 到 v4 的制作说明和校验记录。
- `outputs/assets/`：生成素材原文件、提示词与字体许可证。
- `work/full_rebuild/`：全片渲染及后续修订脚本。最新版为 `revise_full_v4_hand.py`。
- `asset-manifest.json`：原文件大小及 SHA256。上传前逐一核对，未压缩、缩放或重新编码图片。

## 使用

需要 Python 以及 `requirements.txt` 中的包。安装后运行：

```sh
python -m pip install -r requirements.txt
python work/full_rebuild/revise_full_v4_hand.py
```

自行准备有权使用的原视频，放在 `input/source.mp4`，或通过环境变量 `RABBIT_SOURCE` 指定完整路径。可用 `RABBIT_FONT` 指定标题字体。

原视频须与记录对应，渲染脚本依赖它的帧时间戳和音轨。部分镜头保留原视频局部画面。这里未包含原曲、原 MV、官方参考截图或带原曲的最终成片，仓库不能脱离输入视频独立复现完整视频。

公开副本仅做路径适配和隐私清理，已做语法检查；未在公开目录完整重渲染。历史 JSON 是当时本地版本的校验结果，不代表本次重新验收。`outputs/制作说明.md` 对应最早试作，最终版本见 v4 记录。

## 制作过程

先试作 33 到 54 秒片段，核对原片切点与黑场；再扩展全片，尝试三人影子、细线和麻绳弹断。后续减少重复画面，改成六图闪回，最后修正持花手部。各阶段记录保留在 `outputs/`。

## 来源与授权

原曲与 MV：john《rabbit》，参考入口 https://b23.tv/dcxQJNT 。角色参考：https://whitealbum2.jp/character/ 。项目为非官方同人学习记录，与原作者及角色权利方无关联。

代码适用 `LICENSE-CODE`。此许可不涵盖音乐、歌词、角色设计、官方画面或素材。AI 生成角色图在此用于记录制作过程，未对这些图授予统一开放许可，也不表示可自由商用或再分发。字体遵循素材目录内的独立许可证。

## 使用声明

本仓库中的所有代码很有可能并不适用于你需要的场景, 不提供任何技术支持, 所有代码仅供交流学习使用, 使用代码导致的一切后果均由使用者自行承担.

该软件按“原样”提供，不提供任何形式的保证，无论是明示或暗示，包括但不限于适销性、特定目的的适用性和非侵权的保证。在任何情况下，无论是在合同行为、侵权行为或其他情况下，作者或版权所有者均不对任何索赔、损害或其他责任负责，无论是因使用或与软件或软件的其他交易有关的事项而产生或与之相关。

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
